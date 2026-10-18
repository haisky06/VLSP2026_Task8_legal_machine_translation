# Rudeus: VLSP 2026 legal document translation

Two independently fine-tuned VinAI Translate v2 models for English-to-Vietnamese and Vietnamese-to-English legal documents. This repository preparation contains historical pipeline source, Docker serving source and selected evidence. It contains no Qwen training pipeline and no active Modal deployment/management application. The `vinai_modal` folder name is retained for source continuity; users provision their own GPU.

## Results and scope

### Local evaluation on the same 10 bilingual document pairs

Both versions below are evaluated on the same ten held-out public document pairs, in both directions (not ten separate datasets).

| Version | BLEU EN to VI | BLEU VI to EN | Macro BLEU |
|---|---:|---:|---:|
| Earlier refinement baseline ("best60") | 68.8202 | 48.2610 | 58.5406 |
| Selected checkpoint 436 | 69.1865 | 48.4513 | 58.8189 |

The merged stage uses 876 training pairs (793 previous train + 43 former validation + 40 public). The remaining ten public pairs are used for checkpoint selection. These are local validation-selection scores, not an unbiased test estimate. The baseline label "best60" refers to its earlier public50 result, not the ten-pair score above.

### VLSP 2026 competition results reported by AIHub

The team-provided AIHub submission screenshot shows the following scores. The team identifies predictions_3.zip as the earlier version and predictions_4.zip as the newer selected-checkpoint version.

| Version | Submission file | AIHub displayed score | Rounded score | Displayed submission time | Status |
|---|---|---:|---:|---|---|
| Earlier version | `predictions_3.zip` | 61.906902 | 61.9 | 2026-10-14 19:12:31 | Finished |
| Newer version (checkpoint 436) | `predictions_4.zip` | 62.130746 | 62.1 | 2026-10-14 23:30:14 | Finished |

The displayed-score increase is 0.223844 points. Submission times are transcribed as displayed; the screenshot does not identify a timezone. The cropped screenshot does not show the score-column metric name, directional scores, or scoring configuration, so these values are labeled as AIHub displayed scores rather than independently verified directional/macro BLEU. They are separate from the local ten-pair validation scores. No private gold references are included, and these organizer-system scores have not been recomputed locally.

## Historical private pipeline

```text
Source -> source chunks -> VinAI translation -> document assembly
       -> whitespace-only transfer from saved Qwen layout reference
       -> results.csv / prediction ZIP
```

The layout reference is a previously generated translation, not gold data. The formatter preserves VinAI content but does not certify exact input layout. Whitespace-only changes do not automatically guarantee identical BLEU13a tokens (notably `-` immediately before a newline). The team-reported competition scores above are available, but independent local private BLEU/chrF computation requires the unavailable gold references.

**The Docker service returns assembled chunk translations without this document-specific Qwen-reference postformat step.** Its predictions are not claimed byte-identical to the historical formatted private CSV.

## Getting started

1. Use Linux/Python and a suitable CUDA/BF16 GPU environment for model execution. Configure hardware yourself; no cloud GPU is allocated by this repository.
2. Install core dependencies with `python -m pip install -r requirements.txt`. The recorded PyTorch release includes a CUDA 13 runtime; verify driver compatibility rather than assuming all A100 hosts are compatible.
3. Follow [data setup](data/README.md) and [model setup](models/README.md). Model weights, competition datasets and prediction caches are intentionally absent.
4. Run [CPU contract tests](tests/README.md), then follow [training](docs/training.md) or [inference](docs/inference.md).

This is not a one-command fully portable reproduction bundle. Historical scripts retain `/mnt/vinai`, historical job identifiers and some Windows paths. The required state is documented; cloning source alone does not create it. Do not run the historical preparation/selection scripts against an empty directory and assume they discover all missing artifacts.

## Repository map

- `vinai_modal/`: preserved processing/training/evaluation/formatting modules and necessary local compatibility module.
- `docker/`: separate HTTP inference build context; weights must be supplied before building.
- `formatting_reference/`: historical reference and access/license warning; CSV is excluded from Git by default pending publication clearance.
- `docs/`: [model description](docs/model_description.md), [evaluation](docs/evaluation_report.md), [training](docs/training.md), [inference](docs/inference.md), [limitations](docs/limitations.md).
- `results/`: numeric public50 baseline and val10 checkpoint-selection evidence, not full prediction texts.
- `evidence/`: provenance records, selected source snapshots and audits.
- `examples/`: synthetic, non-competition input example.

## Distribution

Do not commit tokens, weights, datasets, optimizer state or Docker archives. The locally supplied Qwen reference contains private-test-derived translations; check redistribution rights before publishing it. No public download URL is fabricated here. Add verified model/Docker download links when distribution permission and access are settled.

The root LICENSE reproduces the upstream VinAI AGPL-3.0 text. See [third-party notices](THIRD_PARTY_NOTICES.md) for scope and attribution. Do not interpret this as relicensing competition data or third-party components.

## Paper

The team will add the paper URL when available. No paper publication is claimed here.
