"""Chunk-based inference matching the selected VinAI decoding settings."""
import gc
import logging
import os
import re
import threading
import time
from pathlib import Path

from .safety import has_letters, repair_translation, repetition_reason, split_source

LANGS={'en-vi':('en_XX','vi_VN'),'vi-en':('vi_VN','en_XX')}
VI_CHARS=re.compile('[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵ]',re.I)

def infer_direction(text,requested):
    if requested in LANGS:return requested
    if requested!='auto':raise ValueError('direction must be en-vi, vi-en, or auto')
    letters=sum(c.isalpha() for c in text)
    if not letters:raise ValueError('Specify direction for non-linguistic input')
    hits=len(VI_CHARS.findall(text))
    return 'vi-en' if hits>=2 and hits/letters>=.005 else 'en-vi'

class Translator:
    def __init__(self):
        self.root=Path(os.environ.get('RUDEUS_MODEL_ROOT','/opt/rudeus/models'))
        self.batch_size=int(os.environ.get('RUDEUS_BATCH_SIZE','32'))
        if not 1<=self.batch_size<=32:raise ValueError('RUDEUS_BATCH_SIZE must be1..32')
        self.lock=threading.Lock();self.model=None;self.tokenizer=None;self.direction=None

    def load(self,direction):
        if self.direction==direction:return
        import torch
        from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
        if not torch.cuda.is_available():raise RuntimeError('A CUDA GPU is required; start Docker with --gpus all')
        if not torch.cuda.is_bf16_supported():raise RuntimeError('A BF16-capable GPU, such as NVIDIA A100, is required')
        self.model=self.tokenizer=None;self.direction=None
        gc.collect();torch.cuda.empty_cache()
        torch.set_num_threads(int(os.environ.get('RUDEUS_THREADS','2')));torch.manual_seed(42)
        path=self.root/direction;source,_=LANGS[direction]
        self.tokenizer=AutoTokenizer.from_pretrained(path,src_lang=source,local_files_only=True)
        self.model=AutoModelForSeq2SeqLM.from_pretrained(path,local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').cuda().eval()
        self.direction=direction
        logging.info('MODEL_LOADED direction=%s gpu=%s',direction,torch.cuda.get_device_name())

    def generate(self,texts,greedy=False):
        import torch
        tok=self.tokenizer
        sequences=[tok.encode(text) for text in texts]
        if any(len(ids)>512 for ids in sequences):raise RuntimeError('Source chunk exceeds512tokens')
        batch=tok.pad({'input_ids':sequences},padding=True,return_tensors='pt')
        batch={k:v.to('cuda') for k,v in batch.items()}
        _,target=LANGS[self.direction]
        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
            output=self.model.generate(**batch,decoder_start_token_id=tok.lang_code_to_id[target],
                num_beams=1 if greedy else 5,do_sample=False,max_new_tokens=1023,use_cache=True,
                repetition_penalty=1.1 if greedy else 1.0)
        output=output.cpu();texts=tok.batch_decode(output,skip_special_tokens=True)
        result=[(text,int(ids.ne(tok.pad_token_id).sum())>=1024) for text,ids in zip(texts,output)]
        del batch,output
        return result

    def translate(self,text,direction='auto'):
        if not text.strip():raise ValueError('Input is empty')
        direction=infer_direction(text,direction)
        with self.lock:
            start=time.monotonic();self.load(direction)
            parts=split_source(text,lambda s:len(self.tokenizer.encode(s)),512)
            assert ''.join(p['raw'] for p in parts)==text
            # Like the original worker, batch longest chunks first, reassemble
            # by original chunk index rather than generation order.
            order=sorted(range(len(parts)),key=lambda i:(-len(self.tokenizer.encode(parts[i]['raw'])),i))
            outputs={};repairs=0
            for begin in range(0,len(order),self.batch_size):
                group=order[begin:begin+self.batch_size]
                initial=self.generate([parts[i]['raw'] for i in group])
                for i,(pred,hit) in zip(group,initial):
                    source=parts[i]['raw']
                    if not has_letters(source):pred=source.strip()
                    elif hit or repetition_reason(source,pred):
                        pred=repair_translation(source,lambda s:self.generate([s])[0],
                            lambda event:logging.info('RETRY direction=%s chunk=%s event=%s',direction,i,event),
                            initial=(pred,hit),fallback=lambda s:self.generate([s],True)[0]).strip()
                        repairs+=1
                    outputs[i]=re.match(r'^\s*',source).group()+pred.strip()+re.search(r'\s*$',source).group()
                logging.info('TRANSLATION direction=%s chunks=%s/%s',direction,min(begin+self.batch_size,len(order)),len(order))
            result=''.join(outputs[i] for i in range(len(parts)))
            if not result.strip():raise RuntimeError('Empty output')
            seconds=time.monotonic()-start
            return result,dict(direction=direction,chunks=len(parts),repairs=repairs,seconds=seconds,
                               words_per_second=len(result.split())/max(seconds,.001))
