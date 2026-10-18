"""Resumeable full fine-tuning of each directional encoder-decoder model."""
import argparse,json,time,hashlib,os,gc,math,random
from pathlib import Path
def main():
    import torch
    from transformers import AutoModelForSeq2SeqLM,AutoTokenizer,Trainer,TrainingArguments,TrainerCallback,DataCollatorForSeq2Seq
    p=argparse.ArgumentParser();p.add_argument('--root',default='/mnt/vinai');p.add_argument('--spec',required=True)
    p.add_argument('--batch',type=int,default=8);p.add_argument('--deadline',type=float,required=True)
    p.add_argument('--output-name',default='training');p.add_argument('--init-name',default=None)
    p.add_argument('--learning-rate',type=float,default=2e-5)
    p.add_argument('--data-dir',default='prepared')
    a=p.parse_args();root=Path(a.root);spec=json.loads(Path(a.spec).read_text());torch.set_num_threads(2)
    class Dataset(torch.utils.data.Dataset):
        def __init__(self,path):self.rows=[json.loads(s) for s in path.read_text().splitlines()]
        def __len__(self):return len(self.rows)
        def __getitem__(self,i):return {k:self.rows[i][k] for k in ('input_ids','attention_mask','labels')}
    class LengthBatches:
        def __init__(self,dataset,batch):
            self.batch_size=batch;self.drop_last=False
            buckets={}
            for i,row in enumerate(dataset.rows):
                length=max(len(row['input_ids']),len(row['labels']))
                assert 0<length<=1024
                bound=next(b for b in (128,256,512,768,1024) if length<=b)
                buckets.setdefault(bound,[]).append(i)
            self.batches=[]
            for bound,members in sorted(buckets.items()):
                members.sort(key=lambda i:(max(len(dataset.rows[i]['input_ids']),len(dataset.rows[i]['labels'])),i))
                self.batches.extend(members[j:j+batch] for j in range(0,len(members),batch))
            random.Random(42).shuffle(self.batches)
            assert sorted(i for group in self.batches for i in group)==list(range(len(dataset)))
        def __len__(self):return len(self.batches)
        def __iter__(self):return iter(self.batches)
    class LengthTrainer(Trainer):
        def get_train_dataloader(self):
            loader=torch.utils.data.DataLoader(self.train_dataset,batch_sampler=LengthBatches(self.train_dataset,a.batch),
                collate_fn=self.data_collator,num_workers=0,pin_memory=True)
            return self.accelerator.prepare(loader)
    for direction in ('en-vi','vi-en'):
        out=root/a.output_name/direction;out.mkdir(parents=True,exist_ok=True)
        if (out/'complete.json').exists():print('Already completed',direction,flush=True);continue
        if time.time()>a.deadline-600:raise RuntimeError('Job deadline; resume remaining direction')
        entry=spec['models'][direction]
        snapshot=entry['snapshot'];initial_sha=None
        if a.init_name:
            previous=root/a.init_name/direction
            if not (previous/'complete.json').is_file():raise RuntimeError('Warm-start requires completed previous training')
            snapshot=str(previous/'model_final')
            initial_sha=hashlib.sha256((previous/'complete.json').read_bytes()).hexdigest()
            for weight in sorted(Path(snapshot).glob('*.safetensors')):
                initial_sha=hashlib.sha256((initial_sha+hashlib.sha256(weight.read_bytes()).hexdigest()).encode()).hexdigest()
        tok=AutoTokenizer.from_pretrained(snapshot,src_lang=entry['source_lang'],local_files_only=True)
        model=AutoModelForSeq2SeqLM.from_pretrained(snapshot,local_files_only=True,dtype=torch.float32,attn_implementation='sdpa')
        model.config.decoder_start_token_id=tok.lang_code_to_id[entry['target_lang']];model.config.use_cache=False
        data=root/a.data_dir/f'{direction}.jsonl'
        plan=dict(model=entry['model'],revision=entry['revision'],data_sha=hashlib.sha256(data.read_bytes()).hexdigest(),
            batch=a.batch,grad_accum=4,epochs=1,lr=a.learning_rate,max_tokens=1024,precision='FP32 master/BF16 autocast',
            init_name=a.init_name,initial_sha=initial_sha,
            batching='length_sorted_buckets_v1',seed=42)
        plan_path=out/'train_plan.json'
        if plan_path.exists() and json.loads(plan_path.read_text())!=plan:raise RuntimeError('Resume plan mismatch')
        plan_path.write_text(json.dumps(plan,indent=2))
        checkpoints=sorted(out.glob('checkpoint-*'),key=lambda x:int(x.name.split('-')[-1]))
        resume=next((x for x in reversed(checkpoints) if all((x/n).exists() for n in ('trainer_state.json','optimizer.pt','scheduler.pt'))),None)
        class Deadline(TrainerCallback):
            def on_save(self,args,state,control,**kwargs):
                print('CHECKPOINT',direction,state.global_step,flush=True)
            def on_step_end(self,args,state,control,**kwargs):
                if time.time()>a.deadline-600:control.should_save=True;control.should_training_stop=True
                return control
        dataset=Dataset(data)
        expected_steps=math.ceil(len(LengthBatches(dataset,a.batch))/4)
        args=TrainingArguments(output_dir=str(out),num_train_epochs=1,per_device_train_batch_size=a.batch,
            gradient_accumulation_steps=4,learning_rate=a.learning_rate,warmup_steps=max(1,int(expected_steps*.03)),lr_scheduler_type='cosine',
            bf16=True,gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
            save_strategy='steps',save_steps=min(200,max(1,expected_steps//4)),save_total_limit=3,logging_steps=20,report_to='none',
            dataloader_num_workers=0,seed=42,data_seed=42,optim='adamw_torch',max_grad_norm=1.)
        trainer=LengthTrainer(model=model,args=args,train_dataset=dataset,processing_class=tok,
            data_collator=DataCollatorForSeq2Seq(tok,model=model,pad_to_multiple_of=8),callbacks=[Deadline()])
        print('TRAIN',direction,'examples',len(trainer.train_dataset),'resume',resume,flush=True)
        trainer.train(resume_from_checkpoint=str(resume) if resume else None)
        if float(trainer.state.epoch or 0)<.999:
            trainer.save_state();raise RuntimeError('Training paused at job deadline; resume checkpoint')
        trainer.save_model(str(out/'model_final'));tok.save_pretrained(str(out/'model_final'))
        (out/'complete.json').write_text(json.dumps(dict(plan=plan,epoch=trainer.state.epoch)))
        del trainer,model;gc.collect();torch.cuda.empty_cache()
if __name__=='__main__':main()
