"""Use old 512-token source chunk inference, never per-line translation."""
import csv,hashlib,json,re,sys,time
from pathlib import Path
from translation_safety import split_source,SAFETY_VERSION

def main():
    from transformers import AutoTokenizer
    root=Path('/mnt/vinai');job=sys.argv[1];phase=sys.argv[2]
    if phase not in ('validation_chunk','private_chunk'):raise ValueError(phase)
    private=phase=='private_chunk';spec=json.loads((root/job/'trained_spec.json').read_text())
    split=json.loads((root/'datasets/merged_public40_split_v1.json').read_text())
    path=root/('private_test_input.csv' if private else 'en-vi-public_test.csv')
    assert hashlib.sha256(path.read_bytes()).hexdigest()==split['private_input_sha' if private else 'public_sha']
    csv.field_size_limit(10000000)
    with path.open(encoding='utf-8-sig',newline='') as f:docs=list(csv.DictReader(f))
    if not private:docs=[d for d in docs if d['sample_id'] in split['validation_ids']]
    assert len(docs)==(100 if private else 10)
    directions={};refs={};expected=[];start=time.monotonic()
    for direction,lang in [('en-vi','en'),('vi-en','vi')]:
        entry=spec['models'][direction];tok=AutoTokenizer.from_pretrained(entry['snapshot'],src_lang=entry['source_lang'],local_files_only=True)
        chunks=[];refs[direction]={}
        for doc in docs:
            if private and doc['source_language']!=lang:continue
            source=doc['source_markdown'] if private else doc['english_markdown' if lang=='en' else 'vietnamese_markdown']
            if not private:refs[direction][doc['sample_id']]=doc['vietnamese_markdown' if lang=='en' else 'english_markdown']
            else:expected.append(dict(sample_id=doc['sample_id'],direction=direction))
            parts=split_source(source,lambda s:len(tok.encode(s)),512)
            assert ''.join(p['raw'] for p in parts)==source
            for index,part in enumerate(parts):
                raw=part['raw'];ids=tok.encode(raw)
                key=hashlib.sha256(json.dumps(dict(id=doc['sample_id'],direction=direction,index=index,source=raw,
                    revision=entry['revision'],beam=5,safety=SAFETY_VERSION),ensure_ascii=False,sort_keys=True).encode()).hexdigest()
                chunks.append(dict(key=key,document_id=doc['sample_id'],index=index,start=part['start'],end=part['end'],raw=raw,
                    boundary=part['boundary'],input_ids=ids,tokens=len(ids),leading=re.match(r'^\s*',raw).group(),trailing=re.search(r'\s*$',raw).group()))
            print('CHUNK_PREP',phase,direction,doc['sample_id'],len(parts),flush=True)
        directions[direction]=chunks
    result=dict(phase='private' if private else 'validation_chunk',directions=directions,references=refs,
                expected_rows=expected,input_sha=hashlib.sha256(path.read_bytes()).hexdigest(),safety_version=SAFETY_VERSION,
                pipeline='original_chunk512_beam5; layout only after private inference',prepare_s=time.monotonic()-start)
    (root/job/f'prepared_{phase}.json').write_text(json.dumps(result,ensure_ascii=False),encoding='utf8')
    print('CHUNK_PREP_COMPLETE',phase,{d:len(v) for d,v in directions.items()},flush=True)

if __name__=='__main__':main()
