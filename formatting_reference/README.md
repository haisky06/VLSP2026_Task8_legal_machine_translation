# Historical private layout reference

`qwen17_private_layout_reference.csv` is a previously generated Qwen3-1.7B prediction file for the 100 historical private documents. It is not organizer gold data and contains no Qwen weights. The corresponding SHA-256 JSON is newly generated packaging metadata from the copied file.

The VinAI formatter matches rows by sample_id and direction. It uses this same-target-language reference for whitespace layout only, retaining VinAI words, numbers, punctuation and order. It cannot format arbitrary new documents lacking corresponding reference rows and does not certify exact source layout.

The archived postformat script expects both reference files under `/mnt/vinai/datasets/`. Place copies there when reproducing the historical run. The local `restore_from_qwen_layout.py` standalone main retains its original Windows paths; use the documented historical postformat entry point rather than assuming its main is portable.

**Publication warning:** this CSV contains full model-generated private-test translations. Its presence in this local preparation folder does not establish permission for public redistribution. It is excluded by the root .gitignore until the team verifies the competition's terms. Do not force-add it without that check. If it cannot be published, distribute it only through an authorized channel and tell users to supply it before historical formatting.
