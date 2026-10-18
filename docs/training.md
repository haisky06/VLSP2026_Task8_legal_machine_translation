# Training the historical VinAI system

## Recorded method

Two full seq2seq models are trained independently, EN to VI first. Historical initial training used 793 accepted pairs for one epoch at 2e-5, followed by one refinement epoch on the same data at 1e-5. The selected merged stage warm-starts from completed `training_refine_v1` weights for one further epoch at 5e-6 on 876 pairs. Microbatch 32, accumulation 4, FP32 master weights/BF16 autocast, length buckets, seed 42 and maximum example side length 1024 are recorded. Step 436 was selected; final completion was step 437.

## Inputs that must be supplied

The original preparation uses legal alignments and split manifests, not a concatenated unfiltered CSV. `prepare_merged_training.py` requires the original train/public CSVs, old manifest, 793 cached alignments, merged split, tokenizer model spec and cached/prepared directories under `/mnt/vinai`. It reuses 793 alignments and adds 83. This repository's evidence metadata does not replace these missing text-bearing state files. See ../data/README.md.

The code retains historical paths and identifiers. Read each script's main function before running. No active Modal app, cloud credentials or scheduler is included. Users choose their GPU/container and provide the expected filesystem. Do not change data membership silently and describe it as the same run.

## Merged-stage trainer invocation

After preparing the exact directional JSONL files and initializing the prior-stage model directories with completion records, the retained trainer can be invoked as follows. Replace placeholders with real paths/state; this is not an automatic downloader:

```bash
python vinai_modal/train.py --root /mnt/vinai \
  --spec /absolute/path/to/model_spec.json \
  --batch 32 --deadline <UNIX_TIMESTAMP> \
  --output-name <RUN_ID>/training \
  --init-name training_refine_v1 --learning-rate 5e-6 \
  --data-dir <RUN_ID>/prepared
```

`--deadline` is a required absolute Unix timestamp, not hours. The code reserves 600 seconds before it and saves/stops incomplete runs. A valid same-stage checkpoint can restore optimizer/scheduler state. An existing completion record causes that direction to be skipped; never point it at unrelated user outputs.

The archived source defaults to a different batch if the argument is omitted. Use the explicitly recorded batch for historical reproduction; choosing another batch creates a different execution configuration. No training was run while packaging this repository.
