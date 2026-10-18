# Rudeus supporting evidence

This bundle accompanies Model_Description.md, Evaluation_Report.md and Formatting_Audit.json. It contains archived metadata and selected source files, not a Docker image or a new training run.

## Reading the evidence

- [Checkpoint comparison and selected scores](02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/checkpoint_selection_report.json)
- [Candidate selection plan](02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/selection_plan.json)
- [Per-document diagnostic scores](02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/selected_document_diagnostics.json)
- [Private inference run metadata](02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/private_chunk/result.json)
- [Original post-format audit](02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/private_formatted/formatting_audit.json)
- [Merged train/validation split](02_VinAI_Checkpoint436_Merged876/data/merged_public40_split_v1.json)
- [Independent metric and file recheck](Independent_Recheck.json)
- [Complete inventory, original SHA-256 hashes and explicit omissions](Evidence_Manifest.json)

## Directory roots

- `02_VinAI_Checkpoint436_Merged876/`: selected system's training plans, selected checkpoint state/configuration, completion records, split/preparation metadata, selection reports, private-run audit and source code.
- `01_VinAI_Best60_Public50/`: clearly labeled earlier-stage train/refinement plans, completion records and historical public50 scores. This is context for weight lineage, not the selected private result.
- `Rudeus_Docker/models/manifest.json`: metadata identifying the packaged step-436 model files; the weight files themselves are in the Docker image, not this evidence directory.

## What is not included

Full training/public/private CSVs, chunk files containing source/reference text, cached full validation translations, raw private translations, the Qwen layout-reference translations and duplicate final prediction CSV/ZIP are deliberately omitted. No access tokens, model weights, optimizer state or Hugging Face caches are included. Document IDs and aggregate/per-document numeric diagnostics remain in the metadata.

The final private prediction CSV/ZIP belongs in the separate private predictions submission folder. Originals of omitted text-bearing files remain in the local archive. Their names and hashes are recorded in the manifest, but a hash is not access to the file and does not reproduce its contents. Full rescoring or inspection of source-to-chunk text requires authorized access to those omitted files. This bundle does not claim complete experimental reproduction by itself.

`Independent_Recheck.json` is a derived record of checks performed on 17 October 2026, not an original training log. Files marked `included_byte_identical` in the manifest are unchanged copies of the original evidence. No original record has been rewritten to remove stale metadata or historical paths.

## Access and verification

Upload the entire `04_Supporting_Materials` directory, including this Evidence directory, to a shared Drive/OneDrive folder. These Markdown links are relative file links, not public web URLs. They work in a downloaded directory with a Markdown viewer. A cloud preview may not resolve them; use the folder tree and paths or download the folder. No upload or cloud link has been created by preparing this local bundle.

The manifest records SHA-256 for each included original file. For example, from this Evidence directory on Linux:

```bash
sha256sum 02_VinAI_Checkpoint436_Merged876/logs/selection/cloud/vinai-select-val10-52a8a667/checkpoint_selection_report.json
```

On Windows PowerShell, use `Get-FileHash -Algorithm SHA256` with the corresponding file path and compare it with the manifest.

## Repository packaging note

The evidence originals above are byte-identical archived copies. Their original file paths and any historical app/resource strings are provenance, not active Modal deployment configuration. No Modal application or management script is shipped in the executable source folder. The local layout-reference CSV is held separately in formatting_reference/ and excluded from Git pending redistribution permission. The originally written omission lists still describe the evidence bundle itself.

See [formatting-reference access](../formatting_reference/README.md), [model documentation](../docs/model_description.md), [evaluation documentation](../docs/evaluation_report.md) and the [root README](../README.md).
