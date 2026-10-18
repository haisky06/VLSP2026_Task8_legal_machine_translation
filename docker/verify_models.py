"""Validate only the local inference payload, without network access."""
import hashlib,json,struct
from pathlib import Path

def main():
    root=Path(__file__).parent/'models'
    data=json.loads((root/'manifest.json').read_text(encoding='utf8'))
    expected={f'{d}/{name}' for d in ('en-vi','vi-en') for name in
              ('config.json','generation_config.json','tokenizer_config.json','tokenizer.json','model.safetensors')}
    if {r['file'] for r in data['files']}!=expected or len(data['files'])!=10:
        raise RuntimeError('Expected exactly10 directional inference files')
    for row in data['files']:
        p=root/row['file'];h=hashlib.sha256()
        with p.open('rb') as f:
            for block in iter(lambda:f.read(4*1024*1024),b''):h.update(block)
        if p.stat().st_size!=row['bytes'] or h.hexdigest()!=row['sha256']:raise RuntimeError('Model checksum mismatch: '+row['file'])
    parameters={}
    for direction in ('en-vi','vi-en'):
        folder=root/direction;cfg=json.loads((folder/'config.json').read_text(encoding='utf8'))
        assert cfg['model_type']=='mbart' and cfg['max_position_embeddings']==1024
        with (folder/'model.safetensors').open('rb') as f:
            n=struct.unpack('<Q',f.read(8))[0];assert 0<n<10000000;header=json.loads(f.read(n))
        tensors={k:v for k,v in header.items() if k!='__metadata__'}
        assert 8+n+max(x['data_offsets'][1] for x in tensors.values())==(folder/'model.safetensors').stat().st_size
        assert all(x['dtype']=='F32' for x in tensors.values())
        import math
        parameters[direction]=sum(math.prod(x['shape']) for x in tensors.values())
        from transformers import AutoTokenizer
        tok=AutoTokenizer.from_pretrained(folder,src_lang='en_XX' if direction=='en-vi' else'vi_VN',local_files_only=True)
        target='vi_VN' if direction=='en-vi' else'en_XX'
        assert tok.lang_code_to_id[target]==cfg['decoder_start_token_id']
    print('Verified10 inference files, SHA256, safetensors structure and offline tokenizer loading.')
    print('Stored tensor elements (includes non-parameter buffers):',parameters)

if __name__=='__main__':main()
