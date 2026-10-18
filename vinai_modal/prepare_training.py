"""Uses existing legal alignments; public and validation never become train examples."""
import csv,json,hashlib,sys,os
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
from collections import Counter
TOK=None
def init(snapshot):
    global TOK
    from transformers import AutoTokenizer
    TOK=AutoTokenizer.from_pretrained(snapshot,src_lang='en_XX',local_files_only=True)
def pack(task):
    row,align=task;out=[];rejected=0;buf=[]
    def encoded(en,vi):
        TOK.src_lang='en_XX';e=TOK.encode(en)
        TOK.src_lang='vi_VN';v=TOK.encode(vi)
        return e,v
    def flush():
        if buf:
            e,v=encoded(''.join(x[0] for x in buf),''.join(x[1] for x in buf))
            if max(len(e),len(v))>1024:raise RuntimeError('Packing exceeded model ceiling')
            out.append(dict(document_id=row['sample_id'],en=e,vi=v));buf.clear()
    previous=None
    for pair in align['pairs']:
        en=row['english_markdown'][pair['en_start']:pair['en_end']]
        vi=row['vietnamese_markdown'][pair['vi_start']:pair['vi_end']]
        if not en.strip() or not vi.strip():continue
        if previous and (previous['en_end']!=pair['en_start'] or previous['vi_end']!=pair['vi_start']):flush()
        e,v=encoded(en,vi)
        if max(len(e),len(v))>1024:flush();rejected+=1;previous=pair;continue
        if buf:
            e2,v2=encoded(''.join(x[0] for x in buf)+en,''.join(x[1] for x in buf)+vi)
            if max(len(e2),len(v2))>512:flush()
        buf.append((en,vi));previous=pair
        if max(len(e),len(v))>512:flush()
    flush();return row['sample_id'],out,rejected
def main():
    root=Path('/mnt/vinai');spec=json.loads(Path(sys.argv[1]).read_text())
    m=json.loads((root/'training_data_manifest.json').read_text())
    assert hashlib.sha256((root/'en-vi-train.csv').read_bytes()).hexdigest()==m['train_sha']
    ids=m['train_ids'];assert not set(ids)&set(m['validation_ids'])
    csv.field_size_limit(sys.maxsize)
    with (root/'en-vi-train.csv').open(encoding='utf-8-sig') as f:rows={r['sample_id']:r for r in csv.DictReader(f)}
    aligned={r['sample_id']:r for r in map(json.loads,(root/'alignment.jsonl').read_text().splitlines())}
    assert set(aligned)==set(ids)
    for sid in ids:
        row,record=rows[sid],aligned[sid]
        for column,key in (('english_markdown','en_sha'),('vietnamese_markdown','vi_sha')):
            assert hashlib.sha256(row[column].encode()).hexdigest()==record[key],f'Stale alignment: {sid} {column}'
        for pair in record['pairs']:
            assert 0<=pair['en_start']<pair['en_end']<=len(row['english_markdown'])
            assert 0<=pair['vi_start']<pair['vi_end']<=len(row['vietnamese_markdown'])
    # Both models use the same bilingual mBART vocabulary; fail rather than silently assume.
    from transformers import AutoTokenizer
    vocab=[AutoTokenizer.from_pretrained(e['snapshot'],src_lang=e['source_lang']).get_vocab() for e in spec['models'].values()]
    assert vocab[0]==vocab[1],'Directional tokenizer vocabularies differ'
    folder=root/'prepared';folder.mkdir(exist_ok=True);docfolder=folder/'documents';docfolder.mkdir(exist_ok=True)
    plan=dict(train_sha=m['train_sha'],split_sha=hashlib.sha256((root/'training_data_manifest.json').read_bytes()).hexdigest(),
        alignment_sha=hashlib.sha256((root/'alignment.jsonl').read_bytes()).hexdigest(),
        revisions={d:e['revision'] for d,e in spec['models'].items()},packing='mbart_contiguous_512_long1024_v1')
    plan_path=folder/'prepare_plan.json'
    if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise RuntimeError('Prepared data plan differs; use a new output')
    plan_path.write_text(json.dumps(plan,indent=2))
    pending=[];result={}
    for sid in ids:
        path=docfolder/(hashlib.sha256(sid.encode()).hexdigest()+'.json')
        if path.exists():result[sid]=json.loads(path.read_text())
        else:pending.append((rows[sid],aligned[sid]))
    with ProcessPoolExecutor(max_workers=4,mp_context=multiprocessing.get_context('spawn'),initializer=init,
        initargs=(spec['models']['en-vi']['snapshot'],)) as pool:
        for sid,chunks,rejected in pool.map(pack,pending,chunksize=1):
            value=dict(id=sid,chunks=chunks,rejected_over_1024=rejected);result[sid]=value
            (docfolder/(hashlib.sha256(sid.encode()).hexdigest()+'.json')).write_text(json.dumps(value))
            print('PREPARE',len(result),'/',len(ids),flush=True)
    count=0
    for direction in ('vi-en','en-vi'):
        temp=folder/(direction+'.tmp')
        with temp.open('w') as f:
            for sid in ids:
                for pair in result[sid]['chunks']:
                    src,tgt=(pair['vi'],pair['en']) if direction=='vi-en' else (pair['en'],pair['vi'])
                    f.write(json.dumps(dict(document_id=sid,input_ids=src,attention_mask=[1]*len(src),labels=tgt))+'\n')
        temp.replace(folder/(direction+'.jsonl'))
    audit=dict(train_documents=len(ids),validation_documents=len(m['validation_ids']),
        chunks=sum(len(x['chunks']) for x in result.values()),rejected_over_1024=sum(x['rejected_over_1024'] for x in result.values()),
        documents_with_chunks=sum(bool(x['chunks']) for x in result.values()),
        train_sha=m['train_sha'],benchmark_spec_sha=hashlib.sha256(Path(sys.argv[1]).read_bytes()).hexdigest())
    (folder/'audit.json').write_text(json.dumps(audit,indent=2));print(audit,flush=True)
if __name__=='__main__':main()
