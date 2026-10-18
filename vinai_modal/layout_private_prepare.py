"""Immutable whitespace/layout template with language-only slots per source line.

Faithful line layout takes priority; this is NOT document-context translation.
"""
import csv,json,re,sys,time,hashlib
from pathlib import Path
from collections import Counter
from translation_safety import SAFETY_VERSION,has_letters

STRUCTURE_VERSION='source_line_cells_v1'

def document_template(source,direction,add_slot):
    template=[]
    def literal(s):
        if s:template.append(dict(literal=s))
    for line in source.splitlines(keepends=True):
        # Protect whitespace separating table columns and indentation, keeping
        # single word spaces inside the text sent to translation.
        for part in re.split(r'([ \t]{2,}|\t|\r\n|\r|\n)',line):
            if not part:continue
            if part.isspace() or not has_letters(part):literal(part);continue
            leading=re.match(r'^\s*',part).group();trailing=re.search(r'\s*$',part).group()
            body=part.strip();literal(leading)
            marker=re.match(r'^(#{1,6}[ \t]+|[-*+][ \t]+|\d+[.)](?:[ \t]+|$)|[a-zđ][.)/](?:[ \t]+|$))',body)
            if marker:literal(marker.group());body=body[marker.end():]
            # Article numbering is syntax and should not be generated anew.
            legal=re.match(r'^(Article|Điều|Chapter|Chương|Section|Mục)([ \t]+)(\d+[a-zA-Z]?|[IVXLCDM]+)([.:-]?[ \t]+|$)',body,re.I)
            if legal:
                mapping={'article':'Điều','chapter':'Chương','section':'Mục'} if direction=='en-vi' else {'điều':'Article','chương':'Chapter','mục':'Section'}
                if legal.group(1).lower() in mapping:
                    literal(mapping[legal.group(1).lower()]+''.join(legal.groups()[1:]));body=body[legal.end():]
            if body:
                if has_letters(body):template.append(dict(slot=add_slot(body)))
                else:literal(body)
            literal(trailing)
    return template

def layout_signature(text):
    return [dict(leading=re.match(r'^[ \t]*',line).group(),trailing=re.search(r'[ \t]*$',line.rstrip('\r\n')).group(),
        blank=not line.strip(),columns=re.findall(r'(?<=\S)[ \t]{2,}(?=\S)',line),
        ending='\r\n' if line.endswith('\r\n') else '\n' if line.endswith('\n') else '\r' if line.endswith('\r') else '')
        for line in text.splitlines(keepends=True)]

def main():
    from transformers import AutoTokenizer
    root=Path('/mnt/vinai');spec=json.loads(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);start=time.monotonic()
    csv.field_size_limit(10000000)
    with (root/'private_test_input.csv').open(encoding='utf-8-sig',newline='') as f:docs=list(csv.DictReader(f))
    assert len(docs)==len({r['sample_id'] for r in docs})==100
    assert Counter(r['source_language'] for r in docs)==Counter({'en':50,'vi':50})
    result=dict(phase='private_layout',structure_version=STRUCTURE_VERSION,directions={},templates={},expected_rows=[],
        input_sha=hashlib.sha256((root/'private_test_input.csv').read_bytes()).hexdigest(),source_signatures={})
    for direction,lang in [('en-vi','en'),('vi-en','vi')]:
        entry=spec['models'][direction];tok=AutoTokenizer.from_pretrained(entry['snapshot'],src_lang=entry['source_lang'],local_files_only=True)
        slots={};rows=[];templates={};signatures={}
        def add_slot(text):
            key=hashlib.sha256((direction+'\0'+text).encode()).hexdigest()
            if key not in slots:
                ids=tok.encode(text)
                if len(ids)>512:raise RuntimeError('Source line exceeds512 tokens; explicit strategy required')
                slots[key]=text;rows.append(dict(key=key,document_id=key,index=0,raw=text,input_ids=ids,tokens=len(ids),leading='',trailing=''))
            return key
        for doc in docs:
            if doc['source_language']!=lang:continue
            sid=doc['sample_id'];templates[sid]=document_template(doc['source_markdown'],direction,add_slot)
            signatures[sid]=layout_signature(doc['source_markdown'])
            result['expected_rows'].append(dict(sample_id=sid,direction=direction))
        result['directions'][direction]=rows;result['templates'][direction]=templates;result['source_signatures'][direction]=signatures
        print('LAYOUT PREP',direction,'documents',len(templates),'unique_slots',len(rows),flush=True)
    result['prepare_s']=time.monotonic()-start;out.parent.mkdir(parents=True,exist_ok=True)
    out.write_text(json.dumps(result,ensure_ascii=False),encoding='utf8');print('LAYOUT PREP COMPLETE',flush=True)

if __name__=='__main__':main()
