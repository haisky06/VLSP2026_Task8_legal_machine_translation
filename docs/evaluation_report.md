# Rudeus: checkpoint-436 evaluation report

Prepared on 17 October 2026. This report separates historical public-test results, held-out checkpoint-selection results and private-file integrity checks. It does not report a private-test BLEU/chrF score, because no private-test gold reference translations were available in the reviewed evidence.

## 1. Evaluation data and supervision disclosure

The merged training set contains 876 bilingual document pairs: 793 previous training pairs, 43 former validation pairs and 40 public-test pairs. Ten of the 50 public-test pairs were held out from this additional supervised training and used for checkpoint selection, in both translation directions.

The ten validation IDs are:

```text
Bat_dong_san/93
Dich_vu_phap_ly/43
Quyen_dan_su/29
Quyen_dan_su/454
Quyen_dan_su/462
So_huu_tri_tue/169
Thu_tuc_to_tung/46
Trach_nhiem_hinh_su/140
Vi_pham_hanh_chinh/20
Vi_pham_hanh_chinh/315
```

Train/validation document IDs are disjoint in the recorded merged split. The private set is not used to select the checkpoint. However, these ten public pairs have been observed during model development and are reused across candidate comparisons. This is a small development/selection set, not an untouched final test set. The inclusion of the other 40 public references in training is disclosed here; this report does not independently certify that this data-use policy satisfies all competition rules.

## 2. Metric calculation and independent recheck

Predictions are reassembled as whole documents before scoring. Corpus BLEU is computed separately for each direction using SacreBLEU `BLEU(tokenize='13a', smooth_method='exp')`. Character F-score uses `CHRF()` with its defaults. Macro is the arithmetic mean of the two directional corpus scores, not an average of document-level BLEU values.

On 17 October, the five candidates' archived predictions were rescored against the same saved ten-pair references. Every directional BLEU/chrF value matched the archived selection report within absolute tolerance 1e-7. This was CPU metric recomputation only: no model was trained or asked to translate again.

The recheck used SacreBLEU 2.6.0, with signatures:

```text
BLEU: nrefs:1|case:mixed|eff:no|tok:13a|smooth:exp|version:2.6.0
chrF: nrefs:1|case:mixed|eff:yes|nc:6|nw:0|space:no|version:2.6.0
```

These are the independently observed recheck signatures, not an assumption about an unrecorded historical package version. Scores below are on the 0–100 scale and rounded to four decimal places.

## 3. Same-ten-pair candidate comparison

| Candidate | BLEU EN to VI | BLEU VI to EN | BLEU macro | chrF EN to VI | chrF VI to EN | chrF macro |
|---|---:|---:|---:|---:|---:|---:|
| Refinement baseline (`best60`) | 68.8202 | 48.2610 | 58.5406 | 82.7398 | 79.4587 | 81.0992 |
| Merged step 327 | 68.8706 | 47.9989 | 58.4348 | 82.7596 | 79.4481 | 81.1038 |
| Merged step 436 | 69.1865 | 48.4513 | 58.8189 | 83.0305 | 79.5501 | 81.2903 |
| Merged step 437 | 69.1754 | 47.9823 | 58.5788 | 83.0130 | 79.3440 | 81.1785 |
| Merged final model (`final876`) | 69.1754 | 47.9823 | 58.5788 | 83.0130 | 79.3440 | 81.1785 |

Selection uses the highest directional corpus BLEU, chrF as a tie-breaker, and deterministic candidate ordering for remaining ties. Step 436 wins for both directions. No weights are averaged or changed by selection.

Relative to the refinement baseline on these same ten pairs, step 436 improves EN-to-VI BLEU by 0.3663, VI-to-EN BLEU by 0.1903, macro BLEU by 0.2783 and macro chrF by 0.1910. These small selection-set differences are not a demonstrated statistically significant improvement or a guarantee of private-test improvement. No confidence interval or significance test is established by the reviewed records.

Steps 437 and `final876` have matching weight hashes for each direction in `selection_plan.json`. Their identical predictions/scores are therefore not two independent experimental confirmations. The selection code reuses the earlier baseline public translations restricted to the ten IDs and reuses the saved merged-final validation translations. Thus, all candidates have the same document IDs, references and metric definitions, but they were not all freshly decoded in one identical batching run. Historical full-public batching and validation-subset batching can have different batch composition. The table is an archived-prediction comparison, not a claim of perfectly controlled fresh decoding for every candidate.

## 4. What the historical 60.09 result means

Before the additional 876-pair stage, the refinement model had these recorded scores on all 50 public pairs:

| Metric | EN to VI | VI to EN | Macro |
|---|---:|---:|---:|
| BLEU | 72.6167 | 47.5561 | 60.0864 |
| chrF | 85.4569 | 79.7046 | 82.5807 |

This result explains the shorthand `best60`. It is not the selected model's private-test score. Comparing its 60.0864 public50 macro BLEU directly against step 436's 58.8189 val10 macro BLEU would mix two evaluation populations. Section 3 uses the common ten-pair subset instead.

For the merged model, the 40 public pairs used for training cannot serve as a held-out test. This report therefore does not present a full-public50 merged score as an unbiased evaluation.

## 5. Document-level diagnostics and remaining weaknesses

The archived `selected_document_diagnostics.json` contains 20 directional document records. Compared with the baseline, 14 document-level BLEU values increase, five decrease and one is unchanged. Corpus BLEU is not the arithmetic mean of these document scores.

Examples that prevent an “improves every document” claim include:

| Document and direction | Baseline document BLEU | Selected document BLEU | Change |
|---|---:|---:|---:|
| `Dich_vu_phap_ly/43`, EN to VI | 69.7902 | 67.9263 | -1.8639 |
| `Thu_tuc_to_tung/46`, VI to EN | 37.3064 | 36.4784 | -0.8280 |
| `Quyen_dan_su/454`, EN to VI | 50.3149 | 51.7385 | +1.4236 |

The selected VI-to-EN corpus BLEU remains 20.7352 points below EN-to-VI on val10. Its chrF gap is smaller, at 3.4804 points. This shows different overlap behavior under the two metrics; it does not by itself identify the cause as alignment, training, decoding or reference wording. No controlled ablation or bilingual legal review establishing that cause is contained in these checks.

Numeric recall in the diagnostic file counts overlapping digit sequences against references. For example, `Thu_tuc_to_tung/46` VI to EN changes from approximately 0.9587 to 0.9421, while its tokenized length ratio changes from 1.0531 to 1.0363. These flags warrant manual review. Number recall ignores context and ordering, and a shorter output is not automatically an omission. Neither flag certifies preservation of legal meaning or correctness of every cited article, date and amount.

## 6. Private inference and file verification

The selected private run completed document assembly for 100 inputs: 50 EN-to-VI and 50 VI-to-EN. Saved input preparation contains 964 and 1,287 source chunks respectively. Independent reassembly of these raw chunks matches the source documents exactly before translation.

The private summary records four repaired chunks, seven retry calls, one initial decoder-limit hit and zero unresolved chunks under `exact_source_retry_v2`. These are operational counts, not translation-quality scores.

Whitespace-only postprocessing was applied after chunk inference. Independent checks on 17 October confirmed:

- exact columns `sample_id,direction,translation`, 100 unique ID/direction keys matching the private input, and no empty translation;
- 50 EN-to-VI rows followed by 50 VI-to-EN rows;
- identical ordered non-whitespace tokens and identical non-whitespace characters before and after formatting for all 100 documents;
- a ZIP containing only root-level `results.csv`, with that CSV byte-identical to the saved final CSV and no ZIP CRC error.

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Final `results.csv` | 5,596,835 | `e15f580d24b3c04bd9eec590c6fa692af34e29639da0f925c6485c264e45b06c` |
| Final `vinai_private_submission.zip` | 1,017,854 | `1dfdac3338e8a03d99ba659877b0189244ae8a7e8ad13ae16a70487080fa2a55` |
| Raw pre-format CSV | 5,405,115 | `5036ac8c7fc0607ec0e1b23f92d94068f3246fb13ec451c05d4bf8a2f559cf51` |

Formatting transfers whitespace cues from a same-target-language Qwen prediction reference, not a private-test gold translation. It is not proof of exact source whitespace, table-cell correspondence or semantic accuracy. “Content unchanged” means unchanged relative to the raw VinAI output, not 100% correct relative to the source or an unavailable private gold reference.

## 7. Metadata and deployment limitations

The split manifest retains a `validation_decode` value of `source_line_cells_exact_layout; same decoding as private`. That field describes an earlier plan and conflicts with the selected run's actual chunk-prepared inputs, source code and `pipeline` value `original_chunk512_beam5; layout only after private inference`. This report describes the latter execution evidence, not the stale split-description field. No original record was edited to remove the discrepancy.

The Docker HTTP service packages step-436 weights but does not include the document-specific Qwen layout reference used to produce this historical formatted CSV. The CSV checks do not certify that a fresh Docker request reproduces its exact bytes. The reviewed build records do not establish full A100 translation validation of the exported service.

## 8. Repository evidence

See the [evidence index](../evidence/README.md) and [inventory](../evidence/Evidence_Manifest.json). Archived metadata is under `evidence/02_VinAI_Checkpoint436_Merged876/` and earlier-stage context under `evidence/01_VinAI_Best60_Public50/`. Historical source snapshots remain identifiable there; runnable-source copies are in `vinai_modal/`. Evaluation summaries are under `results/`.

Full source/reference datasets, cached validation translations, model weights and private prediction packages are not included in this repository. The separate historical Qwen layout-reference CSV is present in the local preparation folder, but excluded from Git by default pending publication permission. A metadata/hash report alone does not enable full rescoring without the omitted texts.

These documents describe the historical selected run, not a claim that a fresh checkout reproduces it with no additional inputs. The Docker service does not implement that run's Qwen-reference postformat step.
