# Third-party notices

## VinAI Translate v2

Directional base models: https://huggingface.co/vinai/vinai-translate-en2vi-v2 and https://huggingface.co/vinai/vinai-translate-vi2en-v2 . Upstream project: https://github.com/VinAIResearch/VinAI_Translate . The upstream AGPL-3.0 license text is retained in LICENSE and docker/LICENSE. Fine-tuned weight manifests record model revisions and hashes; actual weights are not committed. Retain applicable notices when distributing the system.

## LaBSE and alignment

Alignment uses sentence-transformers/LaBSE: https://huggingface.co/sentence-transformers/LaBSE . Recorded revision/settings are in the evidence bundle. The alignment source describes an independent implementation inspired by Bertalign's anchored many-to-many design and DocHPLT reporting principles. Those references and original source comments are preserved; inspiration is not a claim that their implementations were copied.

## Qwen layout reference

formatting_reference/ contains a locally supplied historical Qwen3-1.7B prediction reference used for postprocessing. No Qwen weights or Qwen training code are included. This is generated text derived from competition documents, not an organizer reference translation. Its redistribution is not authorized merely by this notice. See formatting_reference/README.md; the CSV is ignored by Git pending a rights check.

## Software and licensing scope

The project uses PyTorch, Hugging Face Transformers/PEFT-family ecosystem tools where applicable to historical context, Sentence Transformers, SacreBLEU, NumPy and the HTTP service packages listed in dependency files. These retain their own licenses. PEFT is not a requirement of the provided VinAI full-weight training/inference modules. No claim is made that a single root license replaces all dependency licenses.

The root LICENSE is the retained upstream VinAI license text. This preparation does not invent copyright ownership or grant rights over competition data. The team should confirm its own source-distribution permissions before making the repository public. The paper link will be added separately; academic citations do not replace software/model license obligations.
