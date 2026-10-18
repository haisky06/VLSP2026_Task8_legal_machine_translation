"""Compare heldout val10 with fixed chunk decoding; no extra training."""
import hashlib,json,sys,re
from collections import Counter
from pathlib import Path
from sacrebleu.metrics import BLEU,CHRF

ROOT=Path('/mnt/vinai')
SOURCE_JOB='vinai-chunk876-from60-d6738e41'

def load(p):return json.loads(p.read_text())
def hash_file(path):
    h=hashlib.sha256()
    with path.open('rb') as f:
        for part in iter(lambda:f.read(1048576),b''):h.update(part)
    return h.hexdigest()

def score(predictions,refs):
    scores={}
    for direction in ('en-vi','vi-en'):
        ids=sorted(refs[direction])
        if not set(ids)<=set(predictions[direction]):raise RuntimeError('Missing val translations')
        p=[predictions[direction][sid] for sid in ids];r=[refs[direction][sid] for sid in ids]
        scores[direction]=dict(BLEU=BLEU(tokenize='13a',smooth_method='exp').corpus_score(p,[r]).score,
                               chrF=CHRF().corpus_score(p,[r]).score)
    scores['macro']={m:sum(scores[d][m] for d in ('en-vi','vi-en'))/2 for m in ('BLEU','chrF')}
    return scores

def prepare(job):
    folder=ROOT/job;folder.mkdir(parents=True,exist_ok=True)
    spec=load(ROOT/SOURCE_JOB/'trained_spec.json');prepared=load(ROOT/SOURCE_JOB/'prepared_validation_chunk.json')
    split=load(ROOT/'datasets/merged_public40_split_v1.json')
    assert prepared['safety_version']=='exact_source_retry_v2'
    assert prepared['phase']=='validation_chunk'
    assert prepared['input_sha']==split['public_sha']
    assert not set(split['train_ids']) & set(split['validation_ids'])
    assert all(set(prepared['references'][d])==set(split['validation_ids']) for d in ('en-vi','vi-en'))
    source=ROOT/SOURCE_JOB/'training'
    registry={}
    for tag in ('best60','checkpoint327','checkpoint436','checkpoint437','final876'):
        candidate=json.loads(json.dumps(spec))
        for d,entry in candidate['models'].items():
            path=ROOT/'training_refine_v1'/d/'model_final' if tag=='best60' else source/d/('model_final' if tag=='final876' else 'checkpoint-'+tag.removeprefix('checkpoint'))
            if not (path/'config.json').exists() or not list(path.glob('*.safetensors')):raise RuntimeError('Missing candidate '+str(path))
            entry['snapshot']=str(path)
        candidate['model_stage']='val10 checkpoint selection: '+tag
        target=folder/'candidates'/tag;target.mkdir(parents=True,exist_ok=True)
        (target/'spec.json').write_text(json.dumps(candidate,indent=2))
        registry[tag]=dict(spec=str(target/'spec.json'),snapshots={d:e['snapshot'] for d,e in candidate['models'].items()})
    (folder/'prepared_validation_chunk.json').write_text(json.dumps(prepared,ensure_ascii=False))
    ref=prepared['references']
    old=load(ROOT/'vinai-public-retry-2286028317/public_resume/translations.json')
    current=load(ROOT/SOURCE_JOB/'validation_chunk/translations.json')
    result=load(ROOT/SOURCE_JOB/'validation_chunk/result.json')
    final_scores=score(current,ref)
    for d in ('en-vi','vi-en'):
        for m in ('BLEU','chrF'):assert abs(final_scores[d][m]-result['scores'][d][m])<1e-7
    for tag,pred in [('best60',old),('final876',current)]:
        target=folder/'candidates'/tag
        predictions={d:{sid:pred[d][sid] for sid in ref[d]} for d in ('en-vi','vi-en')}
        (target/'translations.json').write_text(json.dumps(predictions,ensure_ascii=False))
        (target/'result.json').write_text(json.dumps(dict(scores=score(pred,ref),documents_per_direction=10,reused_validated=True)))
    # End checkpoint and model_final are usually identical. Prove it using
    # weight byte hashes before reusing metrics instead of another GPU run.
    hashes={};equivalent=True
    for d in ('en-vi','vi-en'):
        weights={tag:{p.name:hash_file(p) for p in Path(registry[tag]['snapshots'][d]).glob('*.safetensors')}
                 for tag in ('checkpoint437','final876')}
        hashes[d]=weights;equivalent &= weights['checkpoint437']==weights['final876']
    if equivalent:
        target=folder/'candidates/checkpoint437'
        (target/'result.json').write_text(json.dumps(dict(scores=final_scores,documents_per_direction=10,
                                                       reused_identical_weights=True,identical_to='final876')))
        (target/'translations.json').write_text((folder/'candidates/final876/translations.json').read_text())
    plan=dict(source_job=SOURCE_JOB,validation_ids=split['validation_ids'],candidates=registry,
              checkpoint437_final_weight_hashes=hashes,checkpoint437_equals_final=equivalent,
              pending=['checkpoint327','checkpoint436']+([] if equivalent else ['checkpoint437']),
              selection_metric='directional BLEU, chrF tie-break, deterministic candidate order',
              private_not_used_to_select=True,no_new_training=True)
    (folder/'selection_plan.json').write_text(json.dumps(plan,indent=2))
    print('SELECTION_PREPARED',json.dumps({k:v for k,v in plan.items() if k not in ('candidates','checkpoint437_final_weight_hashes')}),flush=True)
    return plan

def select(job):
    folder=ROOT/job;plan=load(folder/'selection_plan.json');prepared=load(folder/'prepared_validation_chunk.json')
    results={}
    for tag in plan['candidates']:
        path=folder/'candidates'/tag
        result=load(path/'result.json');pred=load(path/'translations.json')
        calculated=score(pred,prepared['references'])
        for d in ('en-vi','vi-en'):
            for m in ('BLEU','chrF'):assert abs(calculated[d][m]-result['scores'][d][m])<1e-7
        results[tag]=calculated
    selected={d:max(results,key=lambda tag:(results[tag][d]['BLEU'],results[tag][d]['chrF'])) for d in ('en-vi','vi-en')}
    spec=load(ROOT/SOURCE_JOB/'trained_spec.json')
    for d,tag in selected.items():spec['models'][d]['snapshot']=plan['candidates'][tag]['snapshots'][d]
    spec['model_stage']='VinAI val10 selected checkpoints; original chunk512 beam5 safetyv2'
    spec['checkpoint_selection']=selected
    (folder/'trained_spec.json').write_text(json.dumps(spec,indent=2))
    mixed={d:load(folder/'candidates'/selected[d]/'translations.json')[d] for d in selected}
    selected_scores=score(mixed,prepared['references'])
    (folder/'selected_validation_translations.json').write_text(json.dumps(mixed,ensure_ascii=False))
    report=dict(candidates=results,selected=selected,selected_scores=selected_scores,validation_pairs=10,
                no_new_training=True,private_not_used_to_select=True,
                limitation='Validation-selected score, not an unbiased test score; small val and multiple candidates can overfit selection.')
    (folder/'checkpoint_selection_report.json').write_text(json.dumps(report,indent=2))
    diagnostics=[]
    baseline=load(folder/'candidates/best60/translations.json')
    for direction in ('en-vi','vi-en'):
        for sid in sorted(prepared['references'][direction]):
            reference=prepared['references'][direction][sid]
            record=dict(sample_id=sid,direction=direction)
            for label,predictions in [('baseline',baseline),('selected',mixed)]:
                text=predictions[direction][sid]
                b=BLEU(tokenize='13a',smooth_method='exp',effective_order=True).sentence_score(text,[reference])
                nums=Counter(re.findall(r'\d+',reference));out=Counter(re.findall(r'\d+',text))
                record[label]=dict(BLEU=b.score,chrF=CHRF().sentence_score(text,[reference]).score,
                                   length_ratio=b.sys_len/b.ref_len,numeric_recall=sum((nums&out).values())/max(1,sum(nums.values())))
            record['BLEU_delta']=record['selected']['BLEU']-record['baseline']['BLEU']
            diagnostics.append(record)
    diagnostics.sort(key=lambda r:r['BLEU_delta'])
    (folder/'selected_document_diagnostics.json').write_text(json.dumps(dict(documents=diagnostics,
          warning='Number recall flags and score deltas require manual legal-meaning review, not semantic correctness certification.'),indent=2))
    print('CHECKPOINTS_SELECTED',json.dumps(report),flush=True)
    return report

if __name__=='__main__':
    {'prepare':prepare,'select':select}[sys.argv[1]](sys.argv[2])
