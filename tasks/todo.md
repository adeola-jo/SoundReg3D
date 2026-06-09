# tasks/todo.md

## Current: move to notebook-driven exploration (2026-06-09)

- [x] Study `Acoustic-BEV/src/evidential` (feature/mog-slot) code + doc style
- [x] Rewrite `soundreg/` package documentation in that style (tests 10/10)
- [x] Study `acoustic-bev-benchmark` (cleanup) notebook workflow
- [x] Create CLAUDE.md (workflow rules + project-specific rules)
- [x] Commit everything and push to `main` (remote existed: adeola-jo/SoundReg3D)
- [x] Create `exploration` branch
- [x] Write ONE self-contained end-to-end SoundReg notebook
      (notebooks/01_soundreg.ipynb): reuses the MoG-slot front end
      (SpeedFiLM + trunk, faithful to src/evidential network.py), swaps the
      K-slot head for the causal token decoder, trains on the nn_data
      pickles, exports benchmark-format pred CSVs, scores with the
      benchmark matcher (5deg / 5deg+5m gates)
- [x] Smoke-execute the notebook end-to-end before calling it done
      (against contract-exact fake pickles; real data lives on the Linux
      box at /home/joadeola/Datasets/nn_data — the notebook default)
- [x] Push `exploration`

## Review

- Trunk re-implementation checked against network.py source: convs are
  UNPADDED (F: 84 -> 39 -> 18) and conv4 is LazyConv2d — an agent-produced
  shape table (42/21) was wrong; always verify against source.
- STFT channel order confirmed from cut_annotation.py + mic_geometry.yaml:
  interleaved, mic i -> real 2i, imag 2i+1.
- Mic XML axes are (x_fwd, y_up, z_lat); horizontal plane for steering is
  (x, z), matching the Neural-SRP loader's axis swap.
- nn.TransformerDecoder with norm_first needs an explicit final LayerNorm
  and 0.02 embedding init, else tied logits start at CE ~ 100.

## Current block (2026-06-10)

- [x] Audit of the v1 architecture (5 findings; see soundreg_brief.tex
      Section 8 and its Figs. 2-3)
- [x] soundreg_brief.tex updated: Section 4 pointer, new Section 8
      (as built / audit / proposed revision v2), two detailed
      shape-annotated TikZ figures in the Fig. 1 style; compiles
      locally (5 pages); Joseph reviews on Overleaf
- [x] Real data arrived: D:\research-datasets\nn_data (train.pkl 11 GB,
      val, 4 test splits); contract verified (stft (64,84,19),
      velocity dict, ann centers, vehicle.car only)
- [x] Real-data smoke (400 train, 2 epochs): pipeline works, CE
      3.79 -> 3.38, val token acc 0.41, predictions still empty
      (undertrained, immediate EOS as expected)
- [ ] v1 full run (60 epochs, all data) IN FLIGHT: launched 00:16,
      log at %TEMP%\soundreg_run\full.log, outputs to
      notebooks/runs/soundreg_v1 and notebooks/results/. Compare to
      evidential (F1 0.5515 @5deg static_easy, 0.4706 @5deg+5m);
      read the *_together splits and EOS calibration first
- [ ] Implement the v2 fixes in the notebook as Settings-level options
      (attn-pool over T, SRP polar token path, N_THETA=240, MoG
      checkpoint init for the trunk, length-controlled decode report)
- [ ] ORDERING ablation: dominance vs random vs near_to_far

## Known findings to carry into exploration

- Synthetic SRP-peaks baseline: F1 0.35; recall collapses 0.50 -> 0.04 as
  per-source SNR drops below -10 dB, and 0.47 -> 0.20 as concurrency rises
  1 -> 5. The masking phenomenon is present in the testbed.
- SoundReg + spectro_cnn encoder (30 ep, 2000 scenes): token loss learns
  (4.24 -> 3.06) but F1 only 0.04 — the spectrogram CNN gives the decoder
  too little localization evidence at this scale. The polar SRP encoder
  (features on the target grid) is the next thing to try; the notebook
  should default to it.
