"""Best60 warmstart -> train876 -> val10 CHUNKS -> private CHUNKS -> postformat."""
import hashlib, json, os, subprocess, sys, threading, time
from pathlib import Path
import modal

app=modal.App('vlsp-vinai-chunk876-from60')
volume=modal.Volume.from_name('vlsp-vinai',create_if_missing=False)
state=modal.Dict.from_name('vlsp-vinai-jobs',create_if_missing=True)
ROOT=Path('/mnt/vinai');LOCAL=Path(__file__).parent
base=(modal.Image.debian_slim(python_version='3.12')
      .pip_install('torch==2.14.1','transformers==5.17.0','sentencepiece','protobuf','numpy','psutil','nvidia-ml-py','sacrebleu')
      .env({'HF_HOME':'/mnt/vinai/hf_cache','OMP_NUM_THREADS':'2','TOKENIZERS_PARALLELISM':'false'}))
image=base.pip_install('accelerate>=1.10,<2','sentence-transformers').env({'PYTORCH_ALLOC_CONF':'expandable_segments:True'})
for name in ('legal_alignment_core.py','prepare_training.py','prepare_merged_training.py','train.py',
             'layout_private_prepare.py','prepare_merged_layout.py',
             'submission_export.py','public_worker.py','worker.py','public_prepare.py','private_prepare.py',
             'prepare_chunk_eval.py','postformat_chunk_private.py','restore_from_qwen_layout.py'):
    image=image.add_local_file(str(LOCAL/name),'/root/'+name)
image=image.add_local_file(str(LOCAL/'chunk_safety_v2.py'),'/root/translation_safety.py')

def run_command(job,phase,command,timeout):
    folder=ROOT/job;folder.mkdir(parents=True,exist_ok=True);logpath=folder/(phase+'.log')
    state[job]=dict(status=phase,log=str(logpath),direction_order=['en-vi','vi-en'])
    proc=None
    try:
        with logpath.open('a') as log:
            proc=subprocess.Popen(command,stdout=subprocess.PIPE,stderr=subprocess.STDOUT,text=True)
            def consume():
                last=time.monotonic()
                for line in proc.stdout:
                    print(line,end='',flush=True);log.write(line);log.flush()
                    if line.startswith(('PUBLIC_JSON ','MERGE_ALIGN','MERGE_PACK','TRAIN ','CHECKPOINT','PREPARED_LAYOUT')):
                        state[job]=dict(status=phase,progress=line.strip()[:1800],log=str(logpath))
                    if 'CHECKPOINT' in line or time.monotonic()-last>30:
                        volume.commit();last=time.monotonic()
            thread=threading.Thread(target=consume,daemon=True);thread.start()
            try:code=proc.wait(timeout=timeout)
            except subprocess.TimeoutExpired:
                proc.kill();code=proc.wait();raise RuntimeError(phase+' exceeded container stage timeout; caches retained')
            thread.join(timeout=15)
        if code:raise RuntimeError(f'{phase} failed exit={code}; inspect {logpath}')
    finally:
        if proc is not None and proc.poll() is None:proc.kill();proc.wait()
        volume.commit()

@app.function(image=image,cpu=.25,memory=1024,timeout=900,min_containers=0,max_containers=1,retries=0,volumes={'/mnt/vinai':volume})
def preflight(job):
    volume.reload();split=json.loads((ROOT/'datasets/merged_public40_split_v1.json').read_text())
    state[job]=dict(status='preflight',train_documents=876,validation_documents=10)
    def sha(path):
        h=hashlib.sha256()
        with path.open('rb') as f:
            for chunk in iter(lambda:f.read(1048576),b''):h.update(chunk)
        return h.hexdigest()
    ref=ROOT/'datasets/qwen17_private_layout_reference.csv'
    guard=json.loads((ROOT/'datasets/qwen17_private_layout_reference.sha256.json').read_text())
    if sha(ref)!=guard['sha256']:raise RuntimeError('Layout reference hash mismatch')
    for name,key in [('en-vi-train.csv','train_sha'),('en-vi-public_test.csv','public_sha'),
                     ('training_data_manifest.json','previous_manifest_sha'),('private_test_input.csv','private_input_sha')]:
        if sha(ROOT/name)!=split[key]:raise RuntimeError('Cloud data hash mismatch: '+name)
    if len(split['train_ids'])!=876 or len(split['validation_ids'])!=10 or set(split['train_ids']) & set(split['validation_ids']):
        raise RuntimeError('Invalid train/val split')
    for direction in ('en-vi','vi-en'):
        folder=ROOT/'training_refine_v1'/direction
        if not (folder/'complete.json').is_file() or not list((folder/'model_final').glob('*.safetensors')):
            raise RuntimeError('Missing completed model weights: '+direction)
        if float(json.loads((folder/'complete.json').read_text())['epoch'])<.999:
            raise RuntimeError('Source training was partial: '+direction)
    return dict(status='verified',train_documents=876,validation_documents=10,source_model='training_refine_v1')

@app.function(image=image,gpu='RTX-PRO-6000',cpu=(.5,4),memory=(4096,32768),timeout=28800,
              startup_timeout=240,min_containers=0,max_containers=1,single_use_containers=True,retries=0,volumes={'/mnt/vinai':volume})
def train876_from_best60(job):
    volume.reload();split=json.loads((ROOT/'datasets/merged_public40_split_v1.json').read_text())
    for direction in ('en-vi','vi-en'):
        previous=ROOT/'training_refine_v1'/direction
        if not (previous/'complete.json').is_file() or not (previous/'model_final/config.json').is_file():
            raise RuntimeError('Missing completed source weights: '+str(previous))
    import shutil
    origin=ROOT/'vinai-merged876-eaa0a1a0ab/prepared'
    audit=json.loads((origin/'audit.json').read_text())
    if audit['split_sha']!=hashlib.sha256((ROOT/'datasets/merged_public40_split_v1.json').read_bytes()).hexdigest():raise RuntimeError('Prepared split mismatch')
    target=ROOT/job/'prepared';target.mkdir(parents=True,exist_ok=True)
    for name in ('en-vi.jsonl','vi-en.jsonl','audit.json'):
        if not (target/name).exists():shutil.copy2(origin/name,target/name)
    volume.commit()
    run_command(job,'train',[sys.executable,'-u','/root/train.py','--spec','/mnt/vinai/vinai-1b193dc5920e/spec.json',
                           '--batch','32','--deadline',str(time.time()+5*3600),'--output-name',job+'/training',
                           '--init-name','training_refine_v1','--learning-rate','5e-6','--data-dir',job+'/prepared'],18600)
    spec=json.loads((ROOT/'vinai-1b193dc5920e/spec.json').read_text())
    for direction,entry in spec['models'].items():
        previous=ROOT/job/'training'/direction
        if not (previous/'complete.json').exists():raise RuntimeError('Incomplete '+direction)
        entry['snapshot']=str(previous/'model_final')
    spec.update(model_stage='VinAI refine_v1 + 1 additional epoch on 876 documents; public10 held out',
                split_sha=hashlib.sha256((ROOT/'datasets/merged_public40_split_v1.json').read_bytes()).hexdigest(),
                direction_order=['en-vi','vi-en'])
    (ROOT/job/'trained_spec.json').write_text(json.dumps(spec,indent=2));volume.commit()
    return dict(completions={d:json.loads((ROOT/job/'training'/d/'complete.json').read_text()) for d in ('en-vi','vi-en')},
                audit=json.loads((ROOT/job/'prepared/audit.json').read_text()))

@app.function(image=image,cpu=4,memory=(4096,16384),timeout=3600,min_containers=0,max_containers=1,retries=0,volumes={'/mnt/vinai':volume})
def prepare_chunk_inputs(job,phase):
    volume.reload()
    run_command(job,'prepare_'+phase,[sys.executable,'-u','/root/prepare_chunk_eval.py',job,phase],3300)
    return str(ROOT/job/f'prepared_{phase}.json')

@app.function(image=image,gpu='RTX-PRO-6000',cpu=(.5,2),memory=(2048,16384),timeout=28800,
              startup_timeout=240,min_containers=0,max_containers=1,single_use_containers=True,retries=0,volumes={'/mnt/vinai':volume})
def translate_chunks(job,phase,prepared):
    volume.reload();folder=ROOT/job/phase
    run_command(job,phase,[sys.executable,'-u','/root/public_worker.py','--spec',str(ROOT/job/'trained_spec.json'),
                           '--prepared',prepared,'--output',str(folder),'--threads','2'],28200)
    result=json.loads((folder/'result.json').read_text())
    if result['unresolved_chunks']!=0:raise RuntimeError('Unresolved translated slots')
    if phase=='validation_chunk':
        if result['documents_per_direction']!=10:raise RuntimeError('Incomplete val10 evaluation')
    elif result.get('documents')!=100:raise RuntimeError('Private incomplete')
    return result

@app.function(image=image,cpu=2,memory=(2048,8192),timeout=1800,min_containers=0,max_containers=1,retries=0,volumes={'/mnt/vinai':volume})
def postformat_private(job):
    volume.reload()
    run_command(job,'postformat',[sys.executable,'-u','/root/postformat_chunk_private.py',job],1700)
    return json.loads((ROOT/job/'private_formatted/formatting_audit.json').read_text())

@app.function(cpu=.125,memory=256,timeout=86400,min_containers=0,max_containers=1,retries=0)
def driver_train_val10_private_chunks(job):
    try:
        verified=preflight.remote(job)
        training=train876_from_best60.remote(job)
        prepared=prepare_chunk_inputs.remote(job,'validation_chunk')
        validation=translate_chunks.remote(job,'validation_chunk',prepared)
        state[job]=dict(status='validation_completed',scores=validation['scores'])
        prepared=prepare_chunk_inputs.remote(job,'private_chunk')
        private=translate_chunks.remote(job,'private_chunk',prepared)
        formatting=postformat_private.remote(job)
        result=dict(status='completed',preflight=verified,training=training,validation=validation,private=private,formatting=formatting,
                    workspace='thap6753',old_outputs_untouched=True)
        state[job]=result;return result
    except Exception as exc:
        state[job]=dict(status='failed',error=str(exc),job=job,caches_retained=True)
        raise
