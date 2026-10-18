# Limitations and publication checks

- The 10-pair validation set was used to select checkpoints and was repeatedly observed. It is not an unbiased test set. Forty other public pairs enter merged-stage training.
- BLEU/chrF on public50 for the earlier baseline and val10 for step436 are different populations. No private gold score is available in this repository.
- The historical Qwen-reference formatter preserves raw VinAI content, not exact source layout or semantic correctness. The reference itself does not reproduce the input whitespace100%.
- Preserving non-whitespace characters is not sufficient to preserve BLEU13a tokens; its hyphen/newline handling is a known relevant edge case.
- The Docker wrapper omits the document-specific Qwen reference step. Request batching and different GPU/kernel environments can change generation. Full exported-service A100 translation testing has not been established here.
- Historical scripts have fixed `/mnt/vinai`, job IDs, some local Windows paths and required archived state. Source is preserved, not refactored into a fully portable package. Core tests do not prove end-to-end reproducibility.
- No model weights, organizer datasets, optimizer state, full private result CSV/ZIP or full validation caches are committed. The local Qwen reference CSV is excluded from Git until redistribution permission is confirmed.
- LICENSE/third-party notices do not independently authorize publication of competition-derived text. Review rights and personal/service identifiers before making the repo public.
