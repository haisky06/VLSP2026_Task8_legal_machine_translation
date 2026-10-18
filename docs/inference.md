# Inference and historical formatting

## Docker serving

Read ../docker/README.md for service settings and client usage. The Docker build context is `docker/`, not the repo root. Before building, put all files listed by ../models/manifest.json into `docker/models/`, retaining direction subdirectories, and copy that manifest to `docker/models/manifest.json`. This local directory is excluded from Git. The build verifies hashes and tokenizer loading. The context includes its own README and license because the Dockerfile copies them.

```bash
docker build -t rudeus-vinai-legalmt:1.0 docker/
docker run --rm --gpus all -p 8000:8000 --name rudeus-vinai rudeus-vinai-legalmt:1.0
# In a second terminal:
bash docker/client.sh 127.0.0.1:8000 examples/input.md output.md en-vi
```

No cloud hardware is automatically allocated. Models stay FP32 with BF16 autocast; source cap512, beam5, batch32 by default. Users must supply a compatible CUDA GPU/driver. The HTTP service returns document text, not a complete 100-row competition CSV, and does not perform Qwen-reference formatting. Health status is not a translation-quality or A100 end-to-end validation.

## Historical private result pipeline

The selected run's source files are preserved. They expect historical state under `/mnt/vinai`: a `<RUN_ID>/trained_spec.json` pointing to the selected two directional checkpoints, the merged split, source private CSV and prepared chunks. No active Modal wrapper is included.

```bash
python vinai_modal/prepare_chunk_eval.py <RUN_ID> private_chunk
python vinai_modal/public_worker.py \
  --spec /mnt/vinai/<RUN_ID>/trained_spec.json \
  --prepared /mnt/vinai/<RUN_ID>/prepared_private_chunk.json \
  --output /mnt/vinai/<RUN_ID>/private_chunk
python vinai_modal/postformat_chunk_private.py <RUN_ID>
```

The last command also requires `datasets/qwen17_private_layout_reference.csv` and its SHA JSON under `/mnt/vinai`. Supply the exact historical reference only when authorized. The result is written to `<RUN_ID>/private_formatted/results.csv` and its submission ZIP; the raw output is retained separately.

The formatter transfers whitespace from same-language Qwen predictions. It does not load Qwen or replace VinAI words. It is not a general source-only formatter for unseen documents. Whitespace-only editing can still change BLEU13a tokens. Evaluate format changes with the same references/scorer rather than assuming score invariance.

## Compatibility and aliases

The old image mapped chunk_safety_v2.py to translation_safety.py. The new small translation_safety.py module restores that import locally without altering the safety implementation. layout_private_prepare.py is retained solely because public_worker.py has an optional historical layout branch; the selected chunk pipeline does not switch to that branch by including the file. worker.py supplies GPU telemetry and also contains an optional benchmark main; no benchmark is automatically executed.

References/IDs in checkpoint_selection.py retain their historical fixed job paths and require additional archived candidate predictions/reference state to repeat selection. Numeric evidence alone does not fill that state. Do not substitute private predictions or a Qwen reference for organizer gold references.
