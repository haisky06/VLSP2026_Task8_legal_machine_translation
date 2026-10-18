import logging
from fastapi import FastAPI,File,Form,HTTPException,UploadFile
from fastapi.responses import Response
from .inference import Translator

logging.basicConfig(level=logging.INFO,format='%(asctime)s %(levelname)s %(message)s')
app=FastAPI(title='Rudeus VinAI-LegalMT')
translator=Translator()

@app.get('/health')
def health():
    exists=all((translator.root/d/'model.safetensors').is_file() and
               (translator.root/d/'tokenizer.json').is_file() for d in ('en-vi','vi-en'))
    if not exists:raise HTTPException(status_code=503,detail='Model files missing')
    return dict(team='Rudeus',method='VinAI-LegalMT',status='files_available',loaded_direction=translator.direction)

@app.post('/translate')
def translate(file:UploadFile=File(...),direction:str=Form('auto')):
    data=file.file.read(10_000_001)
    if len(data)>10_000_000:raise HTTPException(status_code=413,detail='Maximum input size10MB')
    try:
        text=data.decode('utf-8-sig')
        output,stats=translator.translate(text,direction)
    except (ValueError,UnicodeError) as exc:
        raise HTTPException(status_code=400,detail=str(exc)) from exc
    except Exception:
        logging.exception('Translation failed')
        raise HTTPException(status_code=500,detail='Translation failed; inspect container logs')
    return Response(output,media_type='text/markdown; charset=utf-8',headers={
        'X-VLSP-Direction':stats['direction'],'X-VLSP-Seconds':f'{stats["seconds"]:.3f}',
        'X-VLSP-Chunks':str(stats['chunks']),'X-VLSP-Words-Per-Second':f'{stats["words_per_second"]:.3f}'})
