"""Real public-test inference profiling with resumable translations and corpus metrics."""
import argparse,csv,json,time,os,hashlib,gc,statistics
from pathlib import Path
from collections import defaultdict
from worker import GPUSampler
from translation_safety import SAFETY_VERSION,repetition_reason,repair_translation,has_letters

def emit(x):print('PUBLIC_JSON '+json.dumps(x,ensure_ascii=False),flush=True)
def main():
    import torch,psutil
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    from sacrebleu.metrics import BLEU,CHRF
    p=argparse.ArgumentParser();p.add_argument('--spec',required=True);p.add_argument('--prepared',required=True)
    p.add_argument('--output',required=True);p.add_argument('--probe',action='store_true');p.add_argument('--memory-probe',action='store_true');p.add_argument('--threads',type=int,default=2)
    a=p.parse_args();torch.set_num_threads(a.threads);torch.manual_seed(42)
    spec=json.loads(Path(a.spec).read_text());prepared=json.loads(Path(a.prepared).read_text());profile_only=a.probe or a.memory_probe
    folder=Path(a.output);folder.mkdir(parents=True,exist_ok=True);proc=psutil.Process()
    cpu=lambda:sum(proc.cpu_times()[:2])
    records=[];translations={};gen_total=0;input_total=0;truncated=0;repaired=0;retry_calls=0;unresolved=0
    layout_mode=prepared.get('phase') in ('private_layout','validation_layout')
    for direction in ('en-vi','vi-en'):
        entry=spec['models'][direction];rows=prepared['directions'][direction]
        if a.memory_probe:rows=sorted(rows,key=lambda x:(-x['tokens'],x['key']))[:32]
        elif a.probe:rows=sorted(rows,key=lambda x:(x['tokens'],x['key']))[-32:]
        else:rows=sorted(rows,key=lambda x:(-x['tokens'],x['key']))
        tok=AutoTokenizer.from_pretrained(entry['snapshot'],src_lang=entry['source_lang'],local_files_only=True)
        started=time.monotonic()
        model=AutoModelForSeq2SeqLM.from_pretrained(entry['snapshot'],local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
        emit(dict(kind='load',direction=direction,seconds=time.monotonic()-started,gpu=torch.cuda.get_device_name()))
        start_id=tok.lang_code_to_id[entry['target_lang']]
        cache_path=folder/(direction+'.jsonl');cached={}
        guard=dict(spec_sha=hashlib.sha256(Path(a.spec).read_bytes()).hexdigest(),
            prepared_sha=hashlib.sha256(Path(a.prepared).read_bytes()).hexdigest(),safety_version=SAFETY_VERSION)
        guard_path=folder/(direction+'_cache_guard.json')
        if cache_path.exists() and not profile_only:
            if not guard_path.exists() or json.loads(guard_path.read_text())!=guard:
                raise RuntimeError('Cache model/input/decoding version mismatch; use a new output folder')
        guard_path.write_text(json.dumps(guard))
        if cache_path.exists() and not profile_only:
            for line in cache_path.read_text().splitlines():
                try:r=json.loads(line)
                except json.JSONDecodeError:raise RuntimeError('Interrupted cache line; repair before resuming')
                cached[r['key']]=r
        pending=[x for x in rows if x['key'] not in cached]
        batch_size=128 if layout_mode else 32
        for offset in range(0,len(pending),batch_size):
            group=pending[offset:offset+batch_size]; cpu_before=cpu();total_start=time.monotonic()
            t=time.monotonic();batch=tok.pad({'input_ids':[x['input_ids'] for x in group]},padding=True,return_tensors='pt')
            cpu_prepare_s=time.monotonic()-t
            t=time.monotonic();batch={k:v.to('cuda') for k,v in batch.items()};torch.cuda.synchronize()
            transfer_s=time.monotonic()-t
            if profile_only:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    warm=dict(decoder_start_token_id=start_id,num_beams=5,do_sample=False,
                        max_new_tokens=128 if a.probe else 1023,use_cache=True)
                    if a.probe:warm['min_new_tokens']=128
                    model.generate(**batch,**warm)
                torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats();t=time.monotonic();generation_cpu=cpu()
            with GPUSampler() as telemetry:
                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                    max_new=128 if a.probe else (min(1023,max(128,max(x['tokens'] for x in group)*3+64)) if layout_mode else 1023)
                    kwargs=dict(decoder_start_token_id=start_id,num_beams=5,do_sample=False,
                        max_new_tokens=max_new,use_cache=True)
                    if a.probe:kwargs['min_new_tokens']=128
                    for _ in range(3 if profile_only else 1):
                        output=model.generate(**batch,**kwargs)
                torch.cuda.synchronize()
            generation_s=time.monotonic()-t;generation_cpu_s=cpu()-generation_cpu
            t=time.monotonic();output_cpu=output.cpu();decoded=tok.batch_decode(output_cpu,skip_special_tokens=True)
            decode_s=time.monotonic()-t
            nout=sum(int(r.ne(tok.pad_token_id).sum())-1 for r in output_cpu)*(3 if profile_only else 1)
            nin=sum(x['tokens'] for x in group)*(3 if profile_only else 1);gen_total+=nout;input_total+=nin
            repair_started=time.monotonic();batch_repairs=0
            for x,text,ids in zip(group,decoded,output_cpu):
                flag=int(ids.ne(tok.pad_token_id).sum())>=max_new+1 and not a.probe;truncated+=int(flag)
                initial_flag=flag;reason=None
                if not profile_only:
                    if 'raw' not in x:raise RuntimeError('Prepared input needs raw source; rerun CPU preparation')
                    raw=x['raw']
                    if not has_letters(raw):text=raw.strip();flag=False
                    else:
                        reason='output_limit' if flag else repetition_reason(raw,text)
                        if reason:
                            def generate_retry(piece,greedy=False):
                                nonlocal retry_calls
                                retry_calls+=1;ids_retry=tok.encode(piece)
                                if len(ids_retry)>512:raise RuntimeError('Retry input exceeds source cap')
                                batch_retry=tok.pad({'input_ids':[ids_retry]},padding=True,return_tensors='pt')
                                batch_retry={k:v.to('cuda') for k,v in batch_retry.items()}
                                with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                                    retried=model.generate(**batch_retry,decoder_start_token_id=start_id,
                                        num_beams=1 if greedy else 5,do_sample=False,max_new_tokens=1023,use_cache=True,
                                        repetition_penalty=1.1 if greedy else 1.0)
                                result=tok.decode(retried[0],skip_special_tokens=True)
                                hit=int(retried[0].ne(tok.pad_token_id).sum())>=1024
                                del retried,batch_retry
                                return result,hit
                            def retry_event(event):emit(dict(event,direction=direction,document_id=x['document_id'],chunk=x['index']))
                            text=repair_translation(raw,generate_retry,retry_event,initial=(text,flag),
                                fallback=lambda piece:generate_retry(piece,True)).strip()
                            flag=False;batch_repairs+=1;repaired+=1
                r=dict(key=x['key'],document_id=x['document_id'],index=x['index'],translation=text.strip(),
                    leading=x['leading'],trailing=x['trailing'],output_limit_reached=bool(flag),
                    initial_output_limit_reached=bool(initial_flag),repair_reason=reason,safety_version=SAFETY_VERSION)
                cached[r['key']]=r
                if not profile_only:
                    with cache_path.open('a',encoding='utf-8') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            repair_s=time.monotonic()-repair_started
            row=dict(kind='batch',direction=direction,batch=len(group),completed=min(offset+batch_size,len(pending)),total=len(pending),
                cpu_prepare_s=cpu_prepare_s,transfer_s=transfer_s,generation_s=generation_s,decode_s=decode_s,
                total_s=time.monotonic()-total_start,cpu_seconds=cpu()-cpu_before,generation_cpu_seconds=generation_cpu_s,
                generation_cpu_busy_cores=generation_cpu_s/max(generation_s,.001),input_tokens=nin,output_tokens=nout,
                output_tokens_s=nout/max(generation_s,.001),repair_s=repair_s,repaired_chunks=batch_repairs,
                peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**telemetry.summary())
            records.append(row);emit(row)
            (folder/'profile_partial.json').write_text(json.dumps(records))
            del output,output_cpu,batch;gc.collect();torch.cuda.empty_cache()
        if not profile_only:
            unresolved+=sum(int(r['output_limit_reached']) for r in cached.values())
            documents=defaultdict(list)
            for x in rows:documents[x['document_id']].append(cached[x['key']])
            translations[direction]={sid:''.join(x['leading']+x['translation']+x['trailing']
                for x in sorted(chunks,key=lambda x:x['index'])) for sid,chunks in documents.items()}
        del model;gc.collect();torch.cuda.empty_cache()
    result=dict(records=records,input_tokens=input_total,output_tokens=gen_total,output_limit_reached=truncated,
        repaired_chunks=repaired,retry_calls=retry_calls,safety_version=SAFETY_VERSION,
        throughput_scope='Initial generation only; retry time recorded separately in repair_s',
        unresolved_chunks=unresolved if not profile_only else None,
        cpu_threads=a.threads,probe=profile_only,memory_probe=a.memory_probe,
        allocator=os.environ.get('PYTORCH_ALLOC_CONF','default'),model_stage=spec.get('model_stage','pretrained VinAI Translate v2 before legal fine-tuning'))
    if not profile_only and layout_mode:
        if unresolved:raise RuntimeError('Cannot export unresolved layout translations')
        from submission_export import export_submission
        from layout_private_prepare import layout_signature
        documents={};audit=[]
        for direction,templates in prepared['templates'].items():
            documents[direction]={}
            for sid,template in templates.items():
                segments=[]
                for part in template:
                    if 'literal' in part:segments.append(part['literal'])
                    else:
                        slot=translations[direction][part['slot']]
                        # Whitespace inside a generated slot is not source layout.
                        slot=' '.join(slot.split())
                        if not slot:raise RuntimeError('Empty layout slot')
                        segments.append(slot)
                text=''.join(segments)
                if layout_signature(text)!=prepared['source_signatures'][direction][sid]:
                    raise RuntimeError(f'Layout mismatch: {sid} {direction}')
                documents[direction][sid]=text
                audit.append(dict(sample_id=sid,direction=direction,layout_signature_matches=True,lines=len(text.splitlines())))
        translations=documents
        if prepared['phase']=='private_layout':
            submission=export_submission(folder,documents,prepared['expected_rows'])
            result.update(phase='private_layout',submission=submission,input_sha=prepared['input_sha'],documents=100,
                structure_version=prepared['structure_version'],layout_verified_documents=len(audit))
            emit(dict(kind='private_export',**submission))
        else:
            result.update(phase='validation_layout',input_sha=prepared['input_sha'],
                          structure_version=prepared['structure_version'],layout_verified_documents=len(audit))
        (folder/'layout_audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2),encoding='utf8')
    elif not profile_only and prepared.get('phase')=='private':
        if unresolved:raise RuntimeError('Cannot export unresolved private translations')
        from submission_export import export_submission
        submission=export_submission(folder,translations,prepared['expected_rows'])
        result.update(phase='private',submission=submission,input_sha=prepared['input_sha'],documents=100)
        emit(dict(kind='private_export',**submission))
    if not profile_only and prepared.get('phase') not in ('private','private_layout'):
        scores={};signatures={}
        for direction in ('en-vi','vi-en'):
            docs=prepared['references'][direction]
            assert set(translations[direction])==set(docs),'Incomplete public translations'
            ids=sorted(docs);pred=[translations[direction][sid] for sid in ids];ref=[docs[sid] for sid in ids]
            bleu=BLEU(tokenize='13a',smooth_method='exp');chrf=CHRF()
            scores[direction]={'BLEU':bleu.corpus_score(pred,[ref]).score,'chrF':chrf.corpus_score(pred,[ref]).score}
            signatures[direction]={'BLEU':str(bleu.get_signature()),'chrF':str(chrf.get_signature())}
        scores['macro']={k:statistics.mean(scores[d][k] for d in ('en-vi','vi-en')) for k in ('BLEU','chrF')}
        result.update(scores=scores,signatures=signatures,documents_per_direction=len(prepared['references']['en-vi']))
        (folder/'translations.json').write_text(json.dumps(translations,ensure_ascii=False))
        emit(dict(kind='scores',scores=scores))
    (folder/'result.json').write_text(json.dumps(result,indent=2));emit(dict(kind='complete',probe=a.probe))

if __name__=='__main__':main()
