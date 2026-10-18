# Rudeus — VinAI-LegalMT

Offline English–Vietnamese legal document translation for VLSP2026.

## Models

The system uses two independently fine-tuned encoder–decoder models:

| Direction | Pretrained model | Pretrained revision |
|---|---|---|
| English to Vietnamese | `vinai/vinai-translate-en2vi-v2` | `82f8c91bd22e82085186b45a8a76373f5a79f667` |
| Vietnamese to English | `vinai/vinai-translate-vi2en-v2` | `ae7baa85da07dbe8e23ac26a9f5ef560c17e2138` |

These are full mBART-family model weights, not Qwen models or LoRA adapters. Each model has 12 encoder layers, 12 decoder layers, hidden dimension 1024 and vocabulary size 66773. The included weights are the validation-selected saved models at training step 436 for both directions.

Initial legal-domain training used 793 accepted training document pairs, followed by one refinement epoch on the same data. The refined weights were then trained for one additional epoch on 876 pairs: 793 original training pairs, 43 former validation pairs and 40 public-test pairs. The remaining 10 public-test pairs were held out from that additional training and used to select the saved models. No private-test references were used. This small, repeatedly observed validation set is not an unbiased test set.

## Inference

- Target hardware: one NVIDIA A100 40GB or larger BF16-capable GPU, with NVIDIA Container Toolkit and a driver compatible with the CUDA 13.0 runtime supplied by PyTorch 2.14.1.
- Source chunks: up to 512 tokenizer tokens, preferring sentence/line and then word boundaries. A long indivisible span can require a character-boundary fallback.
- Batch size 32, beam search with 5 beams, no sampling; maximum 1023 new decoder tokens under the model's 1024-position limit.
- FP32 model weights with BF16 autocast, matching the recorded inference method. No quantization or retraining is performed.
- The original `exact_source_retry_v2` checks empty, repetitive and output-limited generations, then retries the same source spans. Unresolved failures return an error rather than a stored prediction.
- One directional model is resident on the GPU at a time. Requests are serialized to prevent model/direction interference.
- All model/tokenizer resources are included locally; inference makes no external service/API calls.

## Build and run

From this directory:

```bash
docker build -t rudeus-vinai-legalmt:1.0 .
docker run --rm --gpus all --network none -p8000:8000 \
  --name rudeus-vinai rudeus-vinai-legalmt:1.0
```

The build installs dependencies and therefore needs Internet. The completed image is intended to run offline. Use a single web worker. `/health` checks file availability, not completion of GPU inference.

## Translate a document

The external client requires Bash and curl. It uploads an UTF-8 text/Markdown file and writes the UTF-8 translation:

```bash
bash client.sh 127.0.0.1:8000 input.md output.md
```

The three arguments are server host:port, input path and output path. An optional fourth argument selects the direction explicitly:

```bash
bash client.sh 127.0.0.1:8000 english.md vietnamese.md en-vi
bash client.sh 127.0.0.1:8000 vietnamese.md english.md vi-en
```

With three arguments, direction is inferred locally from Vietnamese diacritics. This heuristic can be ambiguous for short, unaccented or mixed-language inputs; use the explicit direction in those cases. Runtime, chunk count and words/second are returned in HTTP headers and printed by the client. The maximum request file size is 10MB.

## Output and formatting limitations

Translation is performed on chunks, not independently on every line or table cell. The service rejoins translated chunks in source order and retains boundary whitespace. It does not guarantee exact source line wrapping, indentation or table geometry.

The historical private prediction CSV additionally used a whitespace-only postprocessing step with a same-language Qwen layout reference for those specific documents. That test-specific reference and the precomputed translations are deliberately NOT included in this general-purpose service. This service returns the chunk-assembled translation; it does not claim byte-for-byte reproduction of that formatted CSV. Changing GPU/kernel/batching can also affect numerical decoding. A deployment must be tested before it is described as verified.

## Package contents

`models/en-vi/` and `models/vi-en/` each contain only `model.safetensors`, `config.json`, `generation_config.json`, `tokenizer.json` and `tokenizer_config.json`. `models/manifest.json` records SHA-256 and sizes. `verify_models.py` checks integrity and offline tokenizer loading during the image build.

`app/` contains the inference service and the original retry/splitting helper. No training data, public/private test data, prediction cache, optimizer, scheduler, training logs or access tokens are required or included.

## Upstream attribution and license

The upstream VinAI Translate models are published under GNU AGPL-3.0. The original license text is included in `LICENSE`; this package does not relabel the upstream models as proprietary Rudeus models. Rudeus provides the legal-domain fine-tuning and serving integration. Retain the upstream license and comply with its source/distribution terms when sharing or deploying the system.

Upstream model cards: [EN→VI](https://huggingface.co/vinai/vinai-translate-en2vi-v2), [VI→EN](https://huggingface.co/vinai/vinai-translate-vi2en-v2). Upstream project: [VinAI Translate](https://github.com/VinAIResearch/VinAI_Translate).

Reference: Thien Hai Nguyen et al., *A Vietnamese-English Neural Machine Translation System*, INTERSPEECH2022 Show and Tell.
