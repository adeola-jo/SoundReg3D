# SoundReg Premise Study — Simulation Protocol

**Purpose.** Before building any more architecture, measure whether the premise of SoundReg is true:
that in traffic-like acoustic scenes, (1) quiet sources are often hidden by loud ones, (2) knowing the
loud source helps recover the quiet one, and (3) the 32-mic array physically permits the (r, θ) task.

**What this study decides.** One of four branches, fixed in advance:

| Outcome | Branch |
|---|---|
| Premise holds (P1–P3 pass) | Build SoundReg v2; the theory section writes itself |
| Hiding is rare (P1 fails) | Pivot to ego-noise conditioning (the known masker) |
| Hiding exists but spatial nulling alone fixes it (P2-b) | Pivot to few-microphone regime (NXP-relevant) |
| Physics forbids range (P4 fails) | Rescope: azimuth + near/far classes, fuse with vision |

No training. CPU only. Five working days.

---

## 1. The three quantities, in plain language

### 1.1 Glimpse measure G — "how much of the spectrogram does this source own?"

A source is findable if there are enough time-frequency (TF) cells where it is louder than
everything else combined. For source *j*, given a set *C* of sources we pretend to have
"explained away":

```
G_j(C) = fraction of TF cells where  E_j(t,f)  >  Σ_{i∉C∪{j}} E_i(t,f) + E_noise(t,f)
```

- `G_j(∅)` (C empty) = what a **parallel** detector faces: source j against the full mixture.
- `G_j(louder sources)` = what a **sequential** detector faces after the louder ones are accounted for.
- **Headroom** `H_j = G_j(louder) − G_j(∅)` = how much spectrogram real estate "explaining away"
  returns to source j. If H is near zero everywhere, the whole SoundReg idea has nothing to win.

Computable exactly in simulation because we keep each source's signal separately before mixing.

### 1.2 Oracle subtraction test — "does explaining away actually help a detector?"

The glimpse measure says how much is hidden. This test says whether un-hiding it helps.
Run a **classical, training-free detector** (SRP-PHAT peak picking on the polar grid — code exists
already as the dominance score) under three conditions on the *same* scenes:

- **(a) Plain:** detector sees the full mixture. (= what a parallel model faces)
- **(b) Oracle-SIC:** before looking for source j, subtract the *ground-truth waveforms* of all
  louder sources from the mixture. (= perfect "explaining away"; ceiling for any learned conditioning)
- **(c) Oracle spatial null:** keep the mixture, but apply an MVDR null toward each louder source's
  *known position* (no waveform knowledge). (= what spatial filtering alone can do)

Reading the results:
- (b) ≫ (a) on hidden sources → conditioning has real headroom → **premise holds**.
- (c) ≈ (b) → the array's directionality already solves it; the gain isn't about sequencing,
  it's about spatial resolution → **few-mic branch** (with fewer mics, (c) collapses and sequencing matters).
- (b) ≈ (a) → the information about hidden sources simply isn't recoverable → premise dead.

This single experiment is the model-free version of hypothesis H2 and the most important
measurement in the study.

### 1.3 Cramér–Rao bound (CRB) — "what can this array possibly know?"

For the actual 32-mic geometry (from the XML), compute the best-achievable standard deviation
of azimuth and range estimates for a single source, as a function of position and SNR
(free-field, broadband-stacked narrowband FIM; standard near-field array formulas).

Outputs: two heatmaps over the polar grid, σ_θ(r, θ) and σ_r(r, θ) at reference SNRs
{0, 10, 20 dB}. Decision-relevant numbers: the range r_max beyond which relative range error
exceeds 25%, and how σ_θ compares to the 5° and 1.5° vocabulary bins.

---

## 2. The simulator

### 2.1 Ingredients (all grounded in the existing dataset)

- **Array geometry:** parse the mic XML → 32 positions (already step 1 of the data-prep plan).
- **Source signal bank:** extract windows from the real Acoustic-BEV recordings where `ann`
  contains exactly **one** box → use the beamformed signal toward that box as a quasi-clean
  single-source clip. Target ≥ 200 clips spanning the level range. (Fallback if too few:
  harmonic-plus-broadband synthetic vehicle model; mark clearly as synthetic.)
- **Background/ego-noise bed:** windows with **zero** annotated boxes, grouped by ego speed
  (e.g., 0, 15, 30, 50 km/h).
- **Propagation:** free-field point source → per-mic delay (distance/343 m·s⁻¹) and 1/r amplitude.
  v0 deliberately excludes reflections and car-body shadowing; listed as limitations (§6).
- **Window:** match the model input (the 19-frame STFT window).

### 2.2 Two scene sets

**Set A — Naturalistic (answers "how often"):** 10,000 scenes. N ∈ {1,2,3,4} drawn from the
real dataset's cardinality histogram; positions drawn from the real dataset's (r, θ) distribution;
source levels drawn from the real per-source SNR estimates (data-prep step 7). This set must
mimic reality, because P1 is a claim about reality.

**Set B — Controlled grid (answers "what governs it"):** 2-source scenes sweeping the three
knobs the theory says matter, one at a time:
- angular separation Δθ ∈ {2, 5, 10, 20, 45, 90}°
- level difference ΔSNR ∈ {0, 6, 12, 18, 24} dB
- spectral overlap: same-class pairs (high TF overlap) vs different-class pairs (low overlap)

50 scenes per cell (different clips/positions/noise seeds) → 50 × 6 × 5 × 2 = 3,000 scenes.
Plus the 3-source subset for the order test (P3): 500 scenes, levels staggered 0/−8/−16 dB.

**Stored per scene:** per-source array images (pre-mix), mixture, noise, positions, levels —
so every oracle quantity in §1 is computable exactly.

### 2.3 Detector settings (fixed before running)

SRP-PHAT on the model's own polar grid; peaks = local maxima above a threshold tuned **once**
for best F1 on a 500-scene tuning split of Set A, then frozen. Match to ground truth with the
benchmark gates (5°, 5°+5 m). The tuning split is excluded from all reported numbers.

---

## 3. Measurements

| ID | Quantity | On | Feeds |
|---|---|---|---|
| M1 | G_j(∅) and headroom H_j per source | Set A | P1, Plot 1 |
| M2 | SRP recall per source: plain / oracle-SIC / oracle-null | Sets A & B | P2, Plot 2 |
| M3 | Oracle-SIC recall: dominance order vs reverse vs random (3-source scenes) | Set B-3src | P3 |
| M4 | CRB heatmaps σ_θ, σ_r; r_max(25%); bin-vs-floor comparison | geometry only | P4, Plot 3 |
| M5 | Ego-noise dominance: median G_j of external sources vs ego speed | Set A by speed bin | P5 |
| M6 | Few-mic curve: repeat M2 with subsampled arrays M ∈ {4, 8, 16, 32} | Set B | branch 2 sizing |

M6 is cheap once M2 exists and directly prices the NXP-relevant branch.

---

## 4. Pre-registered decision rules

Numbers below may be debated and edited **before** the first full run; after that they are frozen.
"Hidden" means G_j(∅) < 0.20.

| Rule | Statement | Pass | Fail consequence |
|---|---|---|---|
| **P1** | In Set A scenes with N ≥ 2, fraction of non-loudest sources that are hidden | ≥ 25% | < 10% → hiding is rare → **ego-noise branch** (check P5 first); 10–25% → gray zone: proceed but report |
| **P2-a** | On hidden sources, oracle-SIC recall gain over plain | ≥ +15 pts absolute | < +5 pts → no headroom → premise dead for this data; investigate why before any pivot |
| **P2-b** | Oracle-null recovers ≥ 80% of the oracle-SIC gain | — | If yes → spatial resolution suffices at M=32 → **few-mic branch** via M6 |
| **P3** | Dominance-order oracle-SIC beats reverse order on 3-source scenes (recall of the quietest source) | ≥ +5 pts | < +2 pts → ordering is decoration; keep sequential model but drop order claims |
| **P4** | r_max(25% rel. error) at 10 dB SNR | ≥ 20 m | < 10 m → **rescope branch**: azimuth + near/far; CRB figure becomes a contribution |
| **P5** | Median G of external sources at ≥ 30 km/h | ≥ 0.15 | < 0.15 → ego noise is the elephant masker → ego-conditioning becomes priority regardless of other branches |

Multiple rules can fire; precedence for the *next project step*: P2-a failure > P5 > P2-b > P4 > P1.

---

## 5. The three decision plots

**Plot 1 — "How much hiding is there?"**
CDF of G_j(∅), one curve per loudness rank (1st, 2nd, 3rd loudest), Set A.
Vertical line at G = 0.20. The mass of rank-2/3 curves left of the line *is* the problem
population. Answers P1 at a glance.

**Plot 2 — "Does explaining away help?" (the future Figure 2 of the paper)**
x: G_j(∅) (binned). y: per-source recall. Three curves: plain, oracle-null, oracle-SIC,
with bootstrap CIs (1,000 resamples). The gap between oracle-SIC and plain at low G is the
headroom the learned model will chase; the position of oracle-null between them decides P2-b.

**Plot 3 — "What physics allows."**
σ_θ(r) and σ_r(r) from the CRB at 3 SNRs, with horizontal lines for the 5° and 1.5° bins
(azimuth panel) and the range-bin widths (range panel). Shows in one figure whether the
vocabulary sits above or below the information floor, and where range estimation dies.

Secondary (from Set B): headroom H as a function of Δθ, ΔSNR, and overlap class — the
empirical map of *when* sequencing matters, which later stratifies all model results.

---

## 6. Honest limitations (stated now, so they can't ambush us later)

1. **Free-field propagation** ignores reflections, ground bounce, and car-body shadowing;
   real headroom may differ. Mitigation: §7 real-data anchor.
2. **Quasi-clean clips** are beamformed, not truly isolated; residual interference biases
   G slightly upward (toward optimism about visibility — i.e., against our own hypothesis,
   which is the safe direction).
3. The SRP detector is one classical detector; oracle gains might differ for learned parallel
   models. The study measures *information availability*, not every model's behavior.
4. The CRB is a bound for unbiased estimators; learned models can trade bias for variance.
   It locates the difficulty landscape, not exact model performance.

## 7. Real-data anchor (half a day)

Take real windows with exactly 2 annotated boxes. Estimate each source's SNR by beamforming
(data-prep step 7) and compute an *estimated* G from the beamformed energy ratio. Overlay this
distribution on the simulated Plot 1. If the shapes disagree badly, the simulator's level
statistics get re-fit before any decision rule is read.

## 8. Day-by-day plan

| Day | Work | Output |
|---|---|---|
| 1 | Parse XML; reuse steering-vector code; free-field renderer; unit test: 1 source → SRP peak at truth | working renderer |
| 2 | Clip bank + noise beds from real data; Set A + Set B generation | 13.5k scenes on disk |
| 3 | Glimpse computation (M1, M5); SRP detector + freeze threshold; plain condition | M1 done, Plot 1 draft |
| 4 | Oracle-SIC + oracle-null conditions (M2, M3); few-mic sweep (M6) | Plot 2 draft |
| 5 | CRB derivation + heatmaps (M4); real-data anchor; assemble plots; read decision rules; 2-page memo | decision memo |

## 9. Code layout

```
premise_study/
  geometry.py        # XML → mic positions; steering vectors w(r, θ, f)
  render.py          # source clip + position → per-mic signals (delay, 1/r)
  scenes.py          # Set A / Set B generation; manifest with all ground truth
  glimpse.py         # G_j(C), headroom H_j                     (M1, M5)
  detector.py        # SRP-PHAT peaks; plain / SIC / null modes (M2, M3, M6)
  crb.py             # FIM → σ_θ, σ_r heatmaps                  (M4)
  plots.py           # the three decision plots + Set-B maps
  decide.py          # evaluates P1–P5 against frozen thresholds, prints the branch
```

`decide.py` exists so the conclusion is computed, not narrated.
