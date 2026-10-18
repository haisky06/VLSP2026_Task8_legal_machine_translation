# Model access

Weights are not stored in Git. manifest.json records the expected ten model/tokenizer/config files, directions, step436 and SHA-256 values for the supplied serving models. Acquire the two fine-tuned full model directories through an authorized distribution channel. No public download URL has been verified or invented in this preparation.

Upstream VinAI pretrained models are not replacements for the selected fine-tuned checkpoint. A Qwen LoRA adapter is not a VinAI model. Docker expects `docker/models/en-vi/` and `docker/models/vi-en/` with the manifest at `docker/models/manifest.json`. Historical scripts instead receive model paths via their spec JSON and stored stage directories.

No tokens are required in source. Retain applicable upstream attribution/license and obtain access through a secure mechanism if weights are restricted.
