# Dataset access and historical state

Obtain competition data through the organizer's authorized process; this repository does not redistribute source train/public/private datasets. The private input has no gold translation column. Do not score private predictions against input text or Qwen predictions as if they were gold.

Historical scripts expect `/mnt/vinai/en-vi-train.csv`, `en-vi-public_test.csv`, `private_test_input.csv`, `training_data_manifest.json`, `alignment.jsonl`, `datasets/merged_public40_split_v1.json` and model/preparation state described in docs/training.md. Training data manifest/split copies in evidence can be inspected, but datasets and full cached alignments/prepared examples must be supplied separately. Validate recorded hashes rather than silently mixing versions.
