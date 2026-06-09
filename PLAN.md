# SoundReg — Project Plan

Autoregressive acoustic object detection: from a short multichannel window of one
ego-mounted mic array, emit the set of sounding road users `{(c_i, r_i, θ_i)}` in BEV,
one object at a time, **loudest first** (dominance order), each conditioned on those
already emitted.

This plan is the single source of truth for scope, conventions, and the experiment
schedule. It follows the working brief (`soundreg_brief.pdf`) and adapts the
AutoReg3D formulation (arXiv:2603.07985) from LiDAR near-to-far ordering to
acoustic dominance ordering.

---

## 1. Research question and falsifier

**Hypothesis.** Masking is the acoustic analog of occlusion. Emitting sources in
dominance order and conditioning each prediction on already-emitted sources is an
inductive bias that helps a finite-capacity model recover low-SNR masked sources.

**Headline test (run first).** Recall stratified by per-source SNR and by number of
concurrent sources, SoundReg vs. every parallel baseline.

- *Prediction:* SoundReg's recall advantage **grows** as SNR drops and overlap rises.
- *Falsifier:* a flat advantage. If the gap does not grow at low SNR, the masking
  idea is wrong — stop and rethink.
- *Known opposite failure:* the decoder fires `[EOS]` early and silently drops quiet
  sources → shows up as *worse* low-SNR recall. We therefore always monitor EOS
  calibration and report a length-controlled decoding variant.

**Second prediction (data scaling).** The advantage should *shrink* as training data
grows (inductive bias, not information gain). Data-fraction sweep vs. the query baseline.

---

## 2. What we inherit vs. what is new

| From AutoReg3D (paper) | SoundReg (this project) |
|---|---|
| Objects as short discrete-token sequences, per-attribute vocabularies | Same; object = `(c, t_r, t_θ)` — 2 tokens when single-class |
| Near-to-far ordering (LiDAR occlusion geometry) | **Dominance ordering** by steered-response power (acoustic masking) |
| Point-cloud encoder (voxel/pillar/DSVT) | **Audio encoder**: STFT log-mag + IPD (+ GCC-PHAT, SRP maps) |
| Cartesian (x,y,z,l,w,h,ψ,v) tokens | **Polar BEV** `(r, θ)` tokens; azimuth vocabulary finer than range |
| Single CE loss, teacher forcing, EOS cardinality | Same |
| Greedy / beam / nucleus decoding, cascading refinement | Same, plus sample-S-and-cluster for existence uncertainty |
| GRPO RL fine-tuning (optional later) | Deferred to M6 |

---

## 3. Design principles (research-phase code)

1. **Modular but simple.** Plain PyTorch, no Lightning/Hydra. Small registries +
   one factory wire components from YAML configs. Every swappable axis (encoder,
   decoder behavior, ordering, features, baselines) is a config string.
2. **One interface for all detectors** (`compute_loss(batch)`, `predict(batch)`),
   so SoundReg and every baseline run through the same trainer and the same metrics.
3. **Synthetic-first.** A controllable STFT-domain scene simulator gives exact
   per-source SNR, oracle energies, and NLOS toggles — the whole pipeline runs
   end-to-end before real data arrives, and the inductive-bias experiments are
   cleanest where SNR is controlled.
4. **Everything that matters is logged**: per-GT match flags + SNR (stratified
   recall), EOS probabilities (calibration), per-object scores.

---

## 4. Conventions (fixed project-wide)

- **BEV frame:** ego at origin, x forward, y left. `θ = atan2(y, x)` in **degrees**
  ∈ [-180, 180). `r` in meters. Internally radians only for trig.
- **Polar grid (target vocabulary):** `Nθ` azimuth bins over [-180, 180), `Nr` range
  bins over [0, r_max]. Defaults: Nθ=72 (5°), Nr=24, r_max=30 m. Azimuth finer than
  range, per brief. Detokenization returns bin centers.
- **STFT shapes:** complex `(M, F, T)` internally; stored/real data uses stacked
  real+imag `(2M, F, T)`. Encoder feature images are `(C, T, F)`.
- **Token vocabulary (single shared embedding, per-type offsets):**
  `[PAD=0, BOS=1, EOS=2 | class×C (optional) | range×Nr | azimuth×Nθ]`.
  Object token order: `(class?, range, azimuth)`. EOS is only legal at an object
  boundary (token-type masking enforces this at decode time).
- **Dominance score:** `d_i = Σ_{t,f} |w(r_i,θ_i)^H X(t,f)|²` with delay-and-sum
  steering from the array geometry (near-field, 1/d amplitude).
- **Matching for metrics:** Hungarian on BEV Euclidean distance
  `√(r₁²+r₂²−2r₁r₂cosΔθ)`, valid if ≤ τ_dist (default 2 m) and class agrees.
  Report F1, cardinality error |N̂−N|, range MAE, circular azimuth MAE,
  mean ± std over ≥3 seeds.
- **Fairness rule:** every score/threshold-based baseline gets its operating
  threshold chosen to maximize F1 on **train/val**, never on test (as in the paper).

---

## 5. Repository layout

```
soundreg/
  config.py        YAML configs with _base_ inheritance + dotted CLI overrides
  registry.py      tiny name→class registries
  factory.py       wires grid → tokenizer → steering → dataset → model
  types.py         ScenePrediction / SceneGT containers
  utils.py         seeding, device, params
  data/
    geometry.py    circular array builder + mic-XML parser (adapt in M1)
    polar_grid.py  PolarGrid: quantize/dequantize, polar distance
    steering.py    ArrayManifold (near-field steering), SRP maps, dominance
    features.py    log-mag + IPD stacks; SRP polar feature maps
    ordering.py    ORDERINGS: dominance | near_to_far | azimuth | random | oracle_energy
    tokenizer.py   SceneTokenizer: objects ↔ tokens, type schedule, soft targets
    simulate.py    STFT-domain scene simulator (exact SNR, NLOS flag, oracle energy)
    dataset.py     SyntheticSceneDataset (+ RealArrayDataset skeleton for M1), collate
  models/
    base.py        DetectorBase interface
    pos_enc.py     factorized 2D learned PE (T×F or θ×r "polar PE")
    film.py        SpeedFiLM velocity conditioning (optional)
    encoders/
      cnn.py       SpectroCNNEncoder  (STFT-image → tokens)         [registry: encoder]
      polar.py     PolarSRPEncoder    (SRP polar maps → tokens, circular θ-conv)
    decoder.py     CausalTransformerDecoder (from scratch): causal+context masks,
                   token-type logit masking, greedy/sample/beam, min/max objects
    soundreg.py    SoundRegDetector = encoder + decoder + CE (optional soft targets)
    baselines/
      srp_peaks.py SRP-PHAT peak picking (training-free)            [registry: model]
      heatmap.py   polar heatmap + focal loss + peak extraction
      query_set.py DETR-style query baseline                        [M3]
  eval/metrics.py  Hungarian matching, F1/cardinality/MAE, SNR- and N-stratified
                   recall, threshold sweep, EOS calibration
  training/trainer.py  AdamW + warmup-cosine, AMP, ckpt, JSONL logs
scripts/           train.py, eval.py, sanity_overfit.py
experiments/configs/  base.yaml + per-experiment overrides
tests/             fast end-to-end pipeline tests
```

---

## 6. Milestones

**M0 — Framework + synthetic end-to-end (this commit).**
Everything in §5 except `query_set.py` and real-data ingestion. Acceptance: tests
pass; `sanity_overfit` drives loss down and F1 up on a tiny synthetic set; SRP map
argmax lands on a lone source's bin.

**M1 — Real data ingestion.**
Adapt `geometry.load_mic_xml` to the actual XML; finish `RealArrayDataset` against
the on-hand `items[token]` dict (`stft (64,F,T)`, `ann`, `velocity`, …); record the
real `fs`/frequency axis; precompute per-sample dominance order and beamformed
per-source SNR estimates (stored, eval-only). Decide and freeze the polar grid and
train/val/test split (by scene, not by sample).

**M2 — SoundReg on real data.**
Train the default config; monitor EOS calibration from epoch 1; small sweep of
d_model/layers; pick one "reference model" and freeze it for all comparisons.

**M3 — Baselines (all parallel, same features/encoder budget).**
(a) GCC/SRP peak picking (done in M0, training-free); (b) polar heatmap + peak
extraction (done in M0, needs real-data training); (c) SELD-style class+DOA if
multi-class becomes relevant; (d) DETR-style query-set predictor (`query_set.py`,
reuses the Hungarian matcher; the K-slot MoG ideas from Acoustic-BEV/evidential
are the reference implementation).

**M4 — Headline test + ablations.** In this order:
1. **Headline:** SNR- and overlap-stratified recall, SoundReg vs. each baseline.
   Decision gate: advantage must grow at low SNR, else stop.
2. **Order ablation:** dominance vs near-to-far vs azimuth-sweep vs random vs
   oracle-energy (upper bound). One config flag: `data.ordering=…`.
3. **Conditioning ablation:** full vs context-masked decoder
   (`model.decoder.context_mode=object_only`) — cleanest test that conditioning,
   not tokenization, does the work.
4. **Dominance estimator:** beamformer power vs oracle energy vs SNR (sim only).
5. **Soft target:** one-bin Gaussian (`loss.soft_sigma_bins>0`) vs hard label.
6. **Data scaling:** fraction sweep {10, 25, 50, 100}% vs query baseline —
   advantage should shrink.
7. **EOS diagnostics throughout:** calibration plot + length-controlled decoding
   (`min_objects=N̂_oracle` or quota from val cardinality) reported alongside.

**M5 — NLOS demo (scoped demo, not a benchmark).**
Train with vs without simulated NLOS scenes (simulator flag exists since M0);
compare NLOS detection rate vs the query baseline on the same NLOS targets; small
hand-annotated real NLOS test set if available.

**M6 — Optional extensions.** GRPO RL fine-tuning on F1 reward (per the paper),
cascading refinement (dominance-order prior → random-order completion), sample-S +
clustering for existence uncertainty (sampling already implemented; clustering
utility in eval).

---

## 7. Experiment bookkeeping

- One YAML per experiment in `experiments/configs/`; one-off variations via
  `--set key=value` (both recorded into the run dir).
- Run dir: `runs/<exp_name>/seed<k>/` with `config.yaml`, `log.jsonl`,
  `best.pt`, `metrics.json`, `stratified.npz`.
- Every headline/ablation number = mean ± std over seeds {0,1,2}.

## 8. Risks and mitigations

| Risk | Mitigation |
|---|---|
| EOS fires early, drops quiet sources | EOS calibration monitoring from day one; length-controlled decoding variant always reported |
| Sound-activity labels circular w.r.t. detector | GT annotation must be independent of any detector (brief §5); simulator GT is exact by construction |
| Simulator too easy / unrealistic | It's for pipeline bring-up + controlled inductive-bias studies only; all headline claims need real data (M2+) |
| Angle wrap bugs | Circular Δθ everywhere: `((a−b+180) mod 360) − 180`; unit tests cover wrap |
| Baseline unfairness | Same features, comparable parameter budget, thresholds tuned on val only |
| Tiny sequences make beam/KV-cache moot | Sequences ≤ ~2·N_max+2 tokens; no KV cache needed; revisit only if N grows |
