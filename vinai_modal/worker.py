"""Fixed-shape full-finetuning/inference microbenchmark for both VinAI directions."""
import argparse,json,time,gc,statistics,threading
from pathlib import Path

def emit(row):print('BENCH_JSON '+json.dumps(row),flush=True)

class GPUSampler:
    def __init__(self):
        import pynvml
        self.nv=pynvml;self.nv.nvmlInit();self.handle=self.nv.nvmlDeviceGetHandleByIndex(0)
        self.samples=[];self.stop=threading.Event();self.error=None
    def __enter__(self):
        def run():
            while not self.stop.is_set():
                try:
                    util=self.nv.nvmlDeviceGetUtilizationRates(self.handle)
                    mem=self.nv.nvmlDeviceGetMemoryInfo(self.handle)
                    power=self.nv.nvmlDeviceGetPowerUsage(self.handle)/1000
                    self.samples.append(dict(t=time.monotonic(),gpu_util_percent=util.gpu,
                        memory_util_percent=util.memory,device_memory_used_gib=mem.used/2**30,
                        device_memory_total_gib=mem.total/2**30,power_w=power))
                except Exception as exc:self.error=str(exc)
                self.stop.wait(.1)
        self.thread=threading.Thread(target=run,daemon=True);self.thread.start();return self
    def __exit__(self,*args):
        self.stop.set();self.thread.join(timeout=2);self.nv.nvmlShutdown()
    def summary(self):
        s=self.samples
        if not s:return dict(telemetry_error=self.error or 'No samples')
        return dict(gpu_util_mean_percent=statistics.mean(x['gpu_util_percent'] for x in s),
            gpu_util_peak_percent=max(x['gpu_util_percent'] for x in s),
            memory_activity_util_mean_percent=statistics.mean(x['memory_util_percent'] for x in s),
            nvml_peak_memory_used_gib=max(x['device_memory_used_gib'] for x in s),
            device_memory_total_gib=s[0]['device_memory_total_gib'],
            power_mean_w=statistics.mean(x['power_w'] for x in s),
            power_peak_w=max(x['power_w'] for x in s),telemetry_samples=len(s),telemetry_interval_s=.1,
            telemetry_note='NVML utilization uses device rolling sampling windows; sampled VRAM peak may miss short spikes')

def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer
    p=argparse.ArgumentParser();p.add_argument('--spec');p.add_argument('--deadline',type=float)
    a=p.parse_args();spec=json.loads(Path(a.spec).read_text())
    torch.manual_seed(42);torch.set_num_threads(2)
    for direction in ('vi-en','en-vi'):
        entry=spec['models'][direction]
        if time.time()>a.deadline-60:break
        tokenizer=AutoTokenizer.from_pretrained(entry['snapshot'],src_lang=entry['source_lang'],local_files_only=True)
        # FP32 master weights; BF16 autocast matches ordinary full fine-tuning.
        model=AutoModelForSeq2SeqLM.from_pretrained(entry['snapshot'],local_files_only=True,
            dtype=torch.float32,attn_implementation='sdpa').cuda()
        model.config.decoder_start_token_id=tokenizer.lang_code_to_id[entry['target_lang']]
        model.gradient_checkpointing_enable();model.config.use_cache=False
        optimizer=torch.optim.AdamW(model.parameters(),lr=2e-5)
        emit(dict(kind='model',direction=direction,parameters=sum(p.numel() for p in model.parameters()),
            gpu=torch.cuda.get_device_name(),torch=torch.__version__,revision=entry['revision']))
        for phase in ('train','infer'):
            if phase=='infer':
                optimizer.zero_grad(set_to_none=True);del optimizer;gc.collect();torch.cuda.empty_cache()
                model.gradient_checkpointing_disable();model.config.use_cache=True;model.eval()
            else:model.train()
            for b in (1,2,4,8,16,32,64,128):
                if time.time()>a.deadline-30:
                    emit(dict(kind='case',direction=direction,phase=phase,batch=b,status='skipped_deadline'));continue
                batch=None
                try:
                    ids=torch.tensor([entry['input_ids']]*b,device='cuda')
                    labels=torch.tensor([entry['labels']]*b,device='cuda')
                    batch=dict(input_ids=ids,attention_mask=torch.ones_like(ids))
                    def step():
                        if phase=='train':
                            optimizer.zero_grad(set_to_none=True)
                            with torch.autocast('cuda',dtype=torch.bfloat16):
                                loss=model(**batch,labels=labels,use_cache=False).loss
                            if not torch.isfinite(loss):raise RuntimeError('Non-finite loss')
                            loss.backward();torch.nn.utils.clip_grad_norm_(model.parameters(),1.0);optimizer.step()
                            return b*labels.shape[1]
                        with torch.inference_mode(),torch.autocast('cuda',dtype=torch.bfloat16):
                            out=model.generate(**batch,decoder_start_token_id=model.config.decoder_start_token_id,
                                num_beams=5,do_sample=False,min_new_tokens=128,max_new_tokens=128,use_cache=True)
                        return b*(out.shape[1]-1)
                    torch.cuda.synchronize();start=time.monotonic();step();torch.cuda.synchronize()
                    warmup=time.monotonic()-start;times=[];tokens=0
                    torch.cuda.reset_peak_memory_stats()
                    with GPUSampler() as telemetry:
                        for _ in range(20):
                            if time.time()>a.deadline-10:break
                            torch.cuda.synchronize();start=time.monotonic();tokens+=step();torch.cuda.synchronize()
                            times.append(time.monotonic()-start)
                            # At least 3 repeats and about 3 seconds for utilization sampling.
                            if len(times)>=3 and sum(times)>=3:break
                    if not times:raise TimeoutError('No measured repetitions')
                    speed=tokens/sum(times)
                    emit(dict(kind='case',direction=direction,phase=phase,batch=b,status='ok',
                        source_tokens=512,target_tokens=512 if phase=='train' else 128,beams=5 if phase=='infer' else None,
                        target_or_output_tokens_s=speed,examples_s=b*len(times)/sum(times),
                        median_batch_s=statistics.median(times),repetitions=len(times),warmup_s=warmup,
                        peak_allocated_gib=torch.cuda.max_memory_allocated()/2**30,
                        peak_reserved_gib=torch.cuda.max_memory_reserved()/2**30,**telemetry.summary()))
                except torch.cuda.OutOfMemoryError:
                    emit(dict(kind='case',direction=direction,phase=phase,batch=b,status='oom'));break
                finally:
                    batch=None;ids=None;labels=None
                    if phase=='train':optimizer.zero_grad(set_to_none=True)
                    gc.collect();torch.cuda.empty_cache()
        del model;gc.collect();torch.cuda.empty_cache()

if __name__=='__main__':main()
