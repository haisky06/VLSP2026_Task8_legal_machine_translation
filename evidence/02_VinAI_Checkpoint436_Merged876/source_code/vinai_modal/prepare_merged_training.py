"""Reuse verified old alignments, align only the added 83 document pairs."""
import csv, gc, hashlib, json, os, sys, time
from pathlib import Path
import numpy as np
from legal_alignment_core import align_document, ALIGNMENT_VERSION
from prepare_training import pack, init

def sha(path):return hashlib.sha256(path.read_bytes()).hexdigest()
def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    import torch
    from huggingface_hub import HfApi
    from sentence_transformers import SentenceTransformer
    csv.field_size_limit(10000000)
    root=Path('/mnt/vinai');job=sys.argv[1];folder=root/job/'prepared';folder.mkdir(parents=True,exist_ok=True)
    split=json.loads((root/'datasets/merged_public40_split_v1.json').read_text())
    old=json.loads((root/'training_data_manifest.json').read_text())
    assert sha(root/'en-vi-train.csv')==split['train_sha'] and sha(root/'en-vi-public_test.csv')==split['public_sha']
    assert sha(root/'training_data_manifest.json')==split['previous_manifest_sha']
    assert not set(split['train_ids']) & set(split['validation_ids'])
    rows={r['sample_id']:r for r in read(root/'en-vi-train.csv')+read(root/'en-vi-public_test.csv')}
    legacy={r['sample_id']:r for r in map(json.loads,(root/'alignment.jsonl').read_text().splitlines())}
    assert set(legacy)==set(split['old_train_ids'])
    spec=json.loads((root/'vinai-1b193dc5920e/spec.json').read_text())
    alignment_path=folder/'alignment.jsonl'
    cached={r['sample_id']:r for r in map(json.loads,alignment_path.read_text().splitlines())} if alignment_path.exists() else {}
    assert not set(cached)-set(split['train_ids'])
    for sid,record in legacy.items():
        row=rows[sid]
        for col,key in [('english_markdown','en_sha'),('vietnamese_markdown','vi_sha')]:
            assert hashlib.sha256(row[col].encode()).hexdigest()==record[key],sid
        if sid not in cached:
            with alignment_path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n')
            cached[sid]=record
    missing=[sid for sid in split['train_ids'] if sid not in cached]
    if missing:
        plan_path=folder/'alignment_model.json'
        if plan_path.exists():model_plan=json.loads(plan_path.read_text())
        else:
            model_plan=dict(model='sentence-transformers/LaBSE',revision=HfApi().model_info('sentence-transformers/LaBSE').sha,
                            algorithm=ALIGNMENT_VERSION,threshold=.62,max_group=3)
            plan_path.write_text(json.dumps(model_plan))
        encoder=SentenceTransformer(model_plan['model'],revision=model_plan['revision'],device='cuda',cache_folder=os.environ['HF_HOME'])
        tokenizer=encoder.tokenizer
        def embed(texts):
            pieces=[];groups=[]
            for text in texts:
                ids=tokenizer.encode(text,add_special_tokens=False);members=[]
                for start in range(0,max(1,len(ids)),220):
                    chunk=ids[start:start+220];members.append((len(pieces),max(1,len(chunk))))
                    pieces.append(tokenizer.decode(chunk,skip_special_tokens=True) or ' ')
                groups.append(members)
            vectors=np.asarray(encoder.encode(pieces,batch_size=64,normalize_embeddings=True,convert_to_numpy=True,show_progress_bar=False),dtype=np.float32)
            result=[]
            for members in groups:
                v=np.average(vectors[[i for i,w in members]],axis=0,weights=[w for i,w in members]);v/=max(np.linalg.norm(v),1e-12);result.append(v)
            return np.stack(result)
        for index,sid in enumerate(missing):
            row=rows[sid];started=time.monotonic()
            aligned=align_document(row['english_markdown'],row['vietnamese_markdown'],embed,max_group=3,threshold=.62,max_unit_chars=900)
            record=dict(sample_id=sid,en_sha=hashlib.sha256(row['english_markdown'].encode()).hexdigest(),
                        vi_sha=hashlib.sha256(row['vietnamese_markdown'].encode()).hexdigest(),
                        pairs=[{k:v for k,v in p.items() if k not in ('en','vi')} for p in aligned['pairs']],
                        audit={k:v for k,v in aligned['audit'].items() if k not in ('regions','unmatched_en','unmatched_vi')})
            if not record['pairs']:raise RuntimeError('No accepted alignment for '+sid)
            with alignment_path.open('a') as f:f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush();os.fsync(f.fileno())
            cached[sid]=record
            print('MERGE_ALIGN',index+1,'/',len(missing),sid,'pairs',len(record['pairs']),'coverage',aligned['coverage_en'],aligned['coverage_vi'],'seconds',time.monotonic()-started,flush=True)
        del encoder,tokenizer;gc.collect();torch.cuda.empty_cache()
    assert len(cached)==876
    init(spec['models']['en-vi']['snapshot'])
    docfolder=folder/'documents';docfolder.mkdir(exist_ok=True)
    packed=[];audits=[]
    for i,sid in enumerate(split['train_ids']):
        target=docfolder/(hashlib.sha256(sid.encode()).hexdigest()+'.json')
        if target.exists():record=json.loads(target.read_text())
        else:
            row=rows[sid];alignment=cached[sid]
            for col,key in [('english_markdown','en_sha'),('vietnamese_markdown','vi_sha')]:
                assert hashlib.sha256(row[col].encode()).hexdigest()==alignment[key],sid
            sid,chunks,rejected=pack((row,alignment));record=dict(id=sid,chunks=chunks,rejected_over_1024=rejected)
            if not chunks:raise RuntimeError('No train chunks for '+sid)
            target.write_text(json.dumps(record))
        packed.extend(record['chunks']);audits.append({k:v for k,v in record.items() if k!='chunks'})
        if (i+1)%20==0 or i==875:print('MERGE_PACK',i+1,'/876','chunks',len(packed),flush=True)
    for direction in ('en-vi','vi-en'):
        target=folder/(direction+'.jsonl');tmp=folder/(direction+'.tmp')
        with tmp.open('w') as f:
            for pair in packed:
                src,tgt=(pair['en'],pair['vi']) if direction=='en-vi' else (pair['vi'],pair['en'])
                f.write(json.dumps(dict(document_id=pair['document_id'],input_ids=src,attention_mask=[1]*len(src),labels=tgt))+'\n')
        tmp.replace(target)
    audit=dict(train_documents=876,validation_documents=10,origin_counts=split['origin_counts'],chunks_per_direction=len(packed),
               rejected_over_1024=sum(x['rejected_over_1024'] for x in audits),documents_with_chunks=len(audits),
               split_sha=sha(root/'datasets/merged_public40_split_v1.json'),alignment_sha=sha(alignment_path),
               private_never_train=True,coverage_is_not_precision=True,documents=audits)
    (folder/'audit.json').write_text(json.dumps(audit,indent=2))
    print('MERGE_PREP_COMPLETE',json.dumps({k:v for k,v in audit.items() if k!='documents'}),flush=True)

if __name__=='__main__':main()
