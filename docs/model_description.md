# Rudeus: VinAI-LegalMT model description

Version: validation-selected step 436, both directions. Prepared on 17 October 2026 from archived code, training plans, selection records and prediction files. This document describes the selected VinAI system, not the earlier Qwen systems or an alternative line/cell translation experiment.

## 1. Models and scope

The system uses two independently fine-tuned encoder-decoder models. It does not use a Qwen model or a LoRA adapter for VinAI translation.

| Direction | Pretrained model | Recorded pretrained revision |
|---|---|---|
| English to Vietnamese | `vinai/vinai-translate-en2vi-v2` | `82f8c91bd22e82085186b45a8a76373f5a79f667` |
| Vietnamese to English | `vinai/vinai-translate-vi2en-v2` | `ae7baa85da07dbe8e23ac26a9f5ef560c17e2138` |

The saved configuration specifies `MBartForConditionalGeneration`, 12 encoder layers, 12 decoder layers, hidden dimension 1024, vocabulary size 66,773 and maximum positional embeddings 1024. Fine-tuning updates the full sequence-to-sequence model weights. The pretrained revision identifies the upstream starting model, not the contents of the subsequently fine-tuned checkpoint.

The two selected weight files in the Docker model manifest have these SHA-256 values:

| Direction | SHA-256 of `model.safetensors` |
|---|---|
| EN to VI | `ee5060c25ff558240a5000bee1fdabed7e947977e1bb3538bd4e5cbcb00c1f9b` |
| VI to EN | `3a07e67057b17a726159622f5c22f589f11e836d6a6a7a8ab3556ed6f9f364a0` |

Upstream model cards: [EN to VI](https://huggingface.co/vinai/vinai-translate-en2vi-v2) and [VI to EN](https://huggingface.co/vinai/vinai-translate-vi2en-v2). The submission package retains the upstream AGPL-3.0 license.

## 2. Data and training lineage

The original data manifest records 854 training document pairs and 50 public-test pairs. Eighteen original training IDs were rejected. The remaining 836 pairs were divided into 793 training pairs and 43 validation pairs. Counts here refer to bilingual document pairs, not individual sentences or directional examples.

The recorded weight lineage is:

| Stage | Supervised document pairs | Planned epochs for this stage | Learning rate | Initialization |
|---|---:|---:|---:|---|
| Initial legal-domain fine-tuning | 793 | 1 | 2e-5 | Pretrained directional VinAI models |
| Refinement (`training_refine_v1`) | Same 793 | 1 additional | 1e-5 | Completed initial-stage weights |
| Merged-data fine-tuning | 876 | 1 additional | 5e-6 | Completed refinement weights |

The merged set contains 793 previous training pairs, 43 former validation pairs and 40 public-test pairs. The remaining 10 public-test pairs were held out from merged-data training and used for checkpoint selection. The split is document-level and deterministic with seed 42; its manifest describes category stratification and exclusion of normalized exact duplicates from validation. The checked train-ID and validation-ID sets are disjoint. This is not proof that every possible near-duplicate or legal provision overlap has been eliminated.

The 40 public pairs explicitly enter supervised training using their exposed bilingual references. The former 43-pair validation set is no longer a held-out set for the merged model. Private-test inputs are used for inference, not as supervised training data; no private-test gold reference translations are used in the recorded selection workflow.

The merged run completed at step 437. The selected saved state is step 436, with recorded epoch `0.9988545246277205` for each direction. Therefore, the selected weights are from just before the final update, not literally the final one-epoch state. The label “876” identifies the training set; it does not mean 876 optimizer steps.

Warm-starting this stage loads learned model weights from `training_refine_v1`. It is a new stage with a new optimizer/schedule, not a continuation of the old optimizer state. Within the merged stage, the training code supports resuming its own valid saved optimizer/scheduler checkpoint.

## 3. Alignment and training chunks

The preparation code verifies and reuses the cached alignments for the original 793 training pairs. It aligns only the 83 added pairs: 43 former validation and 40 public pairs. Cached source hashes are checked before reuse.

For the added data, the recorded alignment model is `sentence-transformers/LaBSE`, revision `836121a0533e5664b21c7aacc5d22951f2b8b25b`. The alignment implementation is `qwen_legal_anchors_semantic_dp_v6.0`. Its name is a code-version label; it does not mean that a Qwen generative model performs alignment.

The algorithm partitions legal documents into ordered raw spans, recognizes legal headings, and performs monotone semantic dynamic-programming alignment. Group sizes up to three spans on either side allow symmetric 1:1, 1:n, n:1 and n:m operations within that bound. The recorded threshold is 0.62. This threshold is an algorithmic filter, not a measured 62% correctness probability. Long alignment units are embedded through tokenizer-aware subsegments rather than silently truncating their entire text.

Training packing concatenates consecutive accepted aligned spans only while both language spans remain contiguous. Its usual budget is 512 tokens on each side. An individual aligned pair longer than 512 can be retained if both sides fit the 1024-token ceiling. Pairs exceeding that ceiling would be rejected rather than silently truncated. The merged preparation audit records:

- 876 documents with prepared examples;
- 55,818 examples per direction;
- zero pairs rejected by the over-1024-token rule.

Zero length-rule rejections do not establish complete semantic alignment or 100% source-content coverage. Rejected or unmatched alignment spans are a separate issue. This packing is contiguous bilingual-span packing, not a causal-language-model sequence containing unrelated prompt/answer tasks.

## 4. Merged-stage training configuration

Both directional training plans specify microbatch 32, gradient accumulation 4, one planned additional epoch, learning rate 5e-6, FP32 master weights with BF16 autocast, seed 42 and `length_sorted_buckets_v1` batching. A full accumulation group nominally represents 128 examples on the single GPU; tail groups can be smaller.

The archived trainer uses AdamW, cosine learning-rate scheduling, warmup of approximately 3% of its computed update count, gradient clipping at 1.0 and gradient checkpointing. It pads batches dynamically to a multiple of eight. The merged Modal app requests one RTX PRO 6000 GPU and trains EN to VI before VI to EN. It includes a time/deadline guard that saves progress and stops an incomplete run. The completion records, not the planned epoch setting alone, establish that the recorded merged run finished.

Checkpoint selection is a separate evaluation procedure. It does not update model weights. It compares the refinement baseline, steps 327, 436 and 437, and the final merged model. Directional corpus BLEU is the primary criterion and chrF breaks ties. Step 436 was selected independently for both directions. Evaluation details and the selection limitations are in `Evaluation_Report.md`.

## 5. Recorded chunk inference and retry behavior

The selected validation/private pipeline uses the directional tokenizer to split source documents into chunks of at most 512 tokens. It prefers sentence or line boundaries, then word boundaries. A character-boundary fallback is possible for an otherwise indivisible long span. Saved private chunks were independently checked to reconstruct every source document exactly, before translation.

Initial decoding uses batch size 32, beam size 5, no sampling and up to 1023 new tokens under the model's 1024-position limit. Model weights remain FP32 with BF16 autocast. Chunk batches are length-sorted and outputs are reassembled by original document/chunk order. This pipeline does not prepend an additional chapter/article context prompt from other chunks and does not translate each table cell separately.

`exact_source_retry_v2` checks empty output, certain repeated-phrase or expanded-symbol loops, loss of all alphabetic text, and decoder-limit hits. A flagged span can receive a greedy fallback with repetition penalty 1.1, then recursive regeneration of smaller source spans at safe boundaries, up to the configured depth limit. It does not repair a loop by deleting generated words. Unresolved failures raise an error. These detectors are safeguards, not semantic correctness tests.

The recorded private run used 964 EN-to-VI chunks and 1,287 VI-to-EN chunks. Its summary reports four repaired chunks, seven retry calls, one initial output-limit hit and zero unresolved chunks. “Zero unresolved” means no unresolved condition under these rules, not that all legal meaning, numbers or references are correct.

## 6. Formatting of the historical private prediction ZIP

Translation is completed before layout postprocessing. The formatter transfers whitespace boundaries from a previously generated, same-target-language Qwen 1.7B translation reference. This reference is another model's prediction, not a private-test gold translation. It uses monotone phrase anchors and interpolation, with additional heading/preamble/line-wrap rules. Qwen is not called to regenerate text in this postprocessing step. Its reference supplies layout cues, not replacement VinAI words.

Independent comparison of the saved raw VinAI CSV against the final selected CSV confirms identical ordered whitespace-delimited tokens and identical non-whitespace character sequences for all 100 documents. The final order is 50 EN-to-VI rows followed by 50 VI-to-EN rows. `Formatting_Audit.json` describes the exact checks and hashes.

The result is not certified to match every source space, line break, table cell or semantic layout boundary. Existing translation omissions, repetitions and meaning errors are not corrected by a whitespace-only formatter.

## 7. Docker service versus historical predictions

The exported `rudeus-vinai-legalmt:1.0` image packages the selected step-436 models and chunk/retry service. Its target hardware is an A100 40GB with a BF16-capable, CUDA-compatible driver/runtime setup. The target is not a claim of a completed A100 service benchmark.

The general HTTP service does not include the document-specific Qwen layout reference or stored private predictions. It returns reassembled chunk translations and preserves chunk-boundary whitespace. Consequently, it does not promise byte-identical reproduction of the formatted historical private CSV. Request-level batching and GPU/kernel differences can also affect generation. Image build and model-integrity/tokenizer checks passed; full A100 translation validation of the exported service has not been established in the reviewed records.

## 8. Repository evidence

See the [evidence index](../evidence/README.md) and [inventory](../evidence/Evidence_Manifest.json). Archived metadata is under `evidence/02_VinAI_Checkpoint436_Merged876/` and earlier-stage context under `evidence/01_VinAI_Best60_Public50/`. Historical source snapshots remain identifiable there; runnable-source copies are in `vinai_modal/`. Evaluation summaries are under `results/`.

Full source/reference datasets, cached validation translations, model weights and private prediction packages are not included in this repository. The separate historical Qwen layout-reference CSV is present in the local preparation folder, but excluded from Git by default pending publication permission. A metadata/hash report alone does not enable full rescoring without the omitted texts.

These documents describe the historical selected run, not a claim that a fresh checkout reproduces it with no additional inputs. The Docker service does not implement that run's Qwen-reference postformat step.
