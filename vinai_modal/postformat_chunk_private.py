"""Apply the already-used Qwen layout transfer AFTER raw chunk inference."""
import csv,hashlib,json,re,sys
from pathlib import Path
from restore_from_qwen_layout import transfer,tokenize
from submission_export import export_submission

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))

def main():
    csv.field_size_limit(10000000);root=Path('/mnt/vinai');job=sys.argv[1]
    raw_path=root/job/'private_chunk/results.csv';ref_path=root/'datasets/qwen17_private_layout_reference.csv'
    source_path=root/'private_test_input.csv'
    guard=json.loads((root/'datasets/qwen17_private_layout_reference.sha256.json').read_text())
    assert hashlib.sha256(ref_path.read_bytes()).hexdigest()==guard['sha256']
    before_sha=hashlib.sha256(raw_path.read_bytes()).hexdigest()
    originals={(r['sample_id'],r['direction']):r['translation'] for r in read(raw_path)}
    refs={(r['sample_id'],r['direction']):r['translation'] for r in read(ref_path)}
    assert len(originals)==len(refs)==100 and originals.keys()==refs.keys()
    source=read(source_path);out=root/job/'private_formatted';expected=[];translations={'en-vi':{},'vi-en':{}};audits=[]
    for direction,lang in [('en-vi','en'),('vi-en','vi')]:
        for doc in source:
            if doc['source_language']!=lang:continue
            key=doc['sample_id'],direction;old=originals[key];text,audit=transfer(refs[key],old)
            ow,og=tokenize(old);nw,ng=tokenize(text)
            assert ow==nw and re.sub(r'\s+','',old)==re.sub(r'\s+','',text)
            for a,b in zip(og[1:-1],ng[1:-1]):
                if not re.search(r'[\r\n\t]',a+b) and '  ' not in a+b:assert a==b
            audit.update(sample_id=key[0],direction=direction,content_unchanged=True,
                         source_newlines=doc['source_markdown'].count('\n'))
            audits.append(audit);expected.append(dict(sample_id=key[0],direction=direction));translations[direction][key[0]]=text
    result=export_submission(out,translations,expected)
    assert hashlib.sha256(raw_path.read_bytes()).hexdigest()==before_sha
    reread=read(out/'results.csv')
    for row in reread:assert re.findall(r'\S+',row['translation'])==re.findall(r'\S+',originals[row['sample_id'],row['direction']])
    summary=dict(submission=result,documents=100,content_unchanged_documents=100,raw_translation_untouched=True,
                 raw_csv_sha256=before_sha,reference_sha256=guard['sha256'],method='same Qwen layout transfer used on best60 private output',
                 source_layout_exact=False,limitation='Same-language anchor interpolation is not certification of exact source whitespace/cell semantic correspondence.',
                 row_order='50 EN->VI then 50 VI->EN',audits=audits)
    (out/'formatting_audit.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf8')
    print('POSTFORMAT_COMPLETE',json.dumps({k:v for k,v in summary.items() if k!='audits'}),flush=True)

if __name__=='__main__':main()
