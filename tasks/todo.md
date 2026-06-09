# tasks/todo.md

## Current: move to notebook-driven exploration (2026-06-09)

- [x] Study `Acoustic-BEV/src/evidential` (feature/mog-slot) code + doc style
- [x] Rewrite `soundreg/` package documentation in that style (tests 10/10)
- [x] Study `acoustic-bev-benchmark` (cleanup) notebook workflow
- [x] Create CLAUDE.md (workflow rules + project-specific rules)
- [ ] Commit everything and push to `main` (create GitHub repo if no remote)
- [ ] Create `exploration` branch
- [ ] Write ONE self-contained end-to-end SoundReg notebook
      (Settings -> Data -> Model -> Training -> Curves -> Inference ->
      Evaluation -> Result), modeled on the benchmark notebooks
- [ ] Smoke-execute the notebook end-to-end before calling it done
- [ ] Push `exploration`

## Review

(filled in when the block above completes)

## Known findings to carry into exploration

- Synthetic SRP-peaks baseline: F1 0.35; recall collapses 0.50 -> 0.04 as
  per-source SNR drops below -10 dB, and 0.47 -> 0.20 as concurrency rises
  1 -> 5. The masking phenomenon is present in the testbed.
- SoundReg + spectro_cnn encoder (30 ep, 2000 scenes): token loss learns
  (4.24 -> 3.06) but F1 only 0.04 — the spectrogram CNN gives the decoder
  too little localization evidence at this scale. The polar SRP encoder
  (features on the target grid) is the next thing to try; the notebook
  should default to it.
