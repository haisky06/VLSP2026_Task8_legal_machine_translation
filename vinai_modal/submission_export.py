import csv,io,json,zipfile,hashlib
from pathlib import Path

def export_submission(folder,translations,expected):
    csv.field_size_limit(10000000)
    folder=Path(folder);folder.mkdir(parents=True,exist_ok=True)
    keys=[(r['sample_id'],r['direction']) for r in expected]
    if len(keys)!=100 or len(set(keys))!=100:raise RuntimeError('Expected100 unique ID/direction rows')
    actual={(sid,d) for d,docs in translations.items() for sid in docs}
    if actual!=set(keys):raise RuntimeError('Translation keys do not match private input')
    rows=[]
    for sid,direction in keys:
        text=translations[direction][sid]
        if direction not in ('en-vi','vi-en') or not isinstance(text,str) or not text.strip():raise RuntimeError('Invalid private translation')
        rows.append(dict(sample_id=sid,direction=direction,translation=text))
    buffer=io.StringIO(newline='');writer=csv.DictWriter(buffer,fieldnames=['sample_id','direction','translation'])
    writer.writeheader();writer.writerows(rows);content=buffer.getvalue().encode('utf8')
    reread=list(csv.DictReader(io.StringIO(content.decode('utf8'),newline='')))
    if reread!=rows:raise RuntimeError('CSV round-trip altered translations')
    temp=folder/'results.csv.tmp';temp.write_bytes(content);temp.replace(folder/'results.csv')
    archive=folder/'vinai_private_submission.zip';temp_zip=folder/'submission.zip.tmp'
    with zipfile.ZipFile(temp_zip,'w',compression=zipfile.ZIP_DEFLATED) as z:z.writestr('results.csv',content)
    with zipfile.ZipFile(temp_zip) as z:
        if z.namelist()!=['results.csv'] or z.testzip() is not None or z.read('results.csv')!=content:
            raise RuntimeError('Submission ZIP validation failed')
    temp_zip.replace(archive)
    sha=hashlib.sha256(archive.read_bytes()).hexdigest()
    (folder/'vinai_private_submission.zip.sha256').write_text(sha+'  '+archive.name+'\n')
    return dict(rows=len(rows),columns=['sample_id','direction','translation'],zip=str(archive),sha256=sha,bytes=archive.stat().st_size)
