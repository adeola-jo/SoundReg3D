# SoundReg experiment ledger (exploration branch)

**GOAL (set by Joseph, 2026-06-10): the best single configuration by F1 across
ALL four test sets, target 0.65 to 0.70 F1 on every split, beating the
evidential reference (0.5515 @5deg static_easy). Everything is negotiable:
architecture components (FiLM included), backbone formulation, MAX_OBJECTS,
training recipe, data augmentation (must respect the rig: fixed rooftop array,
driving scenario; superposition-based scene mixing is valid, rotations and
channel permutations are not). Process: diagnose, hypothesize, intervene;
research literature when the evidence points outside the current design.**

One entry per experiment. Single-change discipline: every run differs from its
stated control by exactly one thing, so improvements stay attributable.
Conventions: benchmark matcher, gates 5deg and 5deg+5m; reference rows are
evidential MoG-slot (F1 0.5515 @5deg / 0.4706 @5deg+5m on static_easy).
Artifacts: notebooks/runs/<run>/ (checkpoints, history.csv, curves.png) and
notebooks/results/ (pred + metrics CSVs), both local/gitignored; this ledger is
the committed record. Train data: 4193 scenes (1035 with 0 objects, 2465 with
1, 608 with 2, 85 with 3+); val 280; tests 139/158/277/235 scenes.

---

# MORNING SUMMARY (written 2026-06-10, end of the overnight campaign)

**Where we ended.** Best single configuration: **E4** (v1 architecture +
val-F1 checkpoint selection + early stopping + scene-mixing augmentation).
Checkpoint: notebooks/runs/soundreg_e4_mix/best.pt. The notebook now ships
this recipe as Settings defaults (PATIENCE, MIX_PROB, MIX_GAIN_DB).

F1@5deg per split, all configs (joint-gate numbers in the entries):

| config | st_easy | st_diff | easy_tog | diff_tog | mean |
| --- | --- | --- | --- | --- | --- |
| v1 (CE-selected) | 0.462 | 0.326 | 0.459 | 0.316 | 0.391 |
| E3 selection fix | 0.613 | 0.411 | 0.547 | 0.398 | 0.492 |
| E5 az240 | 0.496 | **0.541** | 0.505 | **0.521** | 0.516 |
| E6 mix+az240 | 0.640 | 0.485 | 0.571 | 0.452 | 0.537 |
| E7 +attn pool | 0.618 | 0.468 | 0.557 | 0.474 | 0.529 |
| E8 deep mix | 0.654 | 0.465 | **0.575** | 0.477 | 0.543 |
| **E4 mix** | **0.655** | 0.497 | 0.551 | 0.472 | **0.544** |
| evidential ref | 0.5515 | | | | |

**Goal status.** SOTA beaten on static_easy at both gates (0.655 vs 0.5515;
0.533 vs 0.4706 joint, the latter from E6). The 0.65 target is met on
static_easy only; difficult splits stand at ~0.50-0.54 best.

**What was established tonight (the science, not just the numbers):**
1. Token CE is a misleading selection signal for this detector (D1): it
   chose checkpoints 0.14-0.19 F1 below the same run's later epochs.
2. The masking/conditioning thesis WORKS: after mixing exposure, masked
   (rank-1+) sources reach recall parity with dominant ones on
   easy_together (D5). This is the brief's core hypothesis, observed.
3. Scene mixing (physically exact superposition) is the strongest single
   lever found: it broke the one-object cardinality cap and lifted recall
   everywhere (E4/E8).
4. FiLM earns its place: removing velocity costs 0.047 F1 on the motion
   split (D5).
5. Near-field misses are emitted-but-5-to-15-degrees-off (D4): largely an
   annotation-center vs acoustic-centroid geometry effect; far misses are
   true low-SNR absences.
6. Rejected with evidence: CE selection (D1), length-controlled decoding
   after mixing (D5), attention pooling over T (E7), confidence filtering
   (D6), mixed-val selection (E9).

**Where the remaining gap to 0.65-everywhere lives.** The difficult splits.
Their losses decompose into (a) near-field angular offsets that the 5 deg
gate punishes (label geometry; possibly recoverable by training the model
to predict annotation centers from acoustic centroids with near-field
oversampling), (b) far/low-SNR true absences, (c) the E5-style
precision-heavy operating point that no mixing config reproduced and that
confidence filtering could not recreate.

**Recommended next steps (in order):**
1. Seed repeats (2 more seeds) of E4 and E8 to put error bars on the
   close calls before believing per-split differences under 0.03.
2. Investigate WHY E5 (az240, no mixing) wins difficult splits with high
   precision: compare its predictions to E4's on the same scenes
   (which detections differ?); if it is a real mechanism, a curriculum
   (train az240 first, add mixing later) may capture both.
3. Near-field: oversample 0-10 m scenes or weight their loss; check
   whether the model can learn the centroid-to-center offset.
4. The SRP polar token path (v2 fix 2) remains untested: highest-value
   untried architecture change, especially for range (the joint gate).
5. Range vocabulary and range errors have had no attention at all yet;
   the joint-gate numbers (mADE ~1.7-1.9 m) suggest headroom.

---

## E1: v1 baseline (control)

Run `soundreg_v1`, 2026-06-10 00:36. Notebook as committed: reused MoG-slot
front end (random init), mean over T, 360 angular tokens, decoder d=192 x4,
vocab 24 r x 72 az, greedy decode, 60 epochs.

| split | gate | P | R | F1 | mAAE | mADE |
| --- | --- | --- | --- | --- | --- | --- |
| static_easy | 5deg | 0.531 | 0.410 | 0.462 | 2.42 | |
| static_easy | 5deg+5m | 0.494 | 0.381 | 0.430 | 2.45 | 1.79 |
| static_difficult | 5deg | 0.465 | 0.251 | 0.326 | 2.32 | |
| static_difficult | 5deg+5m | 0.343 | 0.186 | 0.241 | 2.36 | 1.76 |
| easy_together | 5deg | 0.517 | 0.413 | 0.459 | 2.30 | |
| easy_together | 5deg+5m | 0.461 | 0.368 | 0.409 | 2.28 | 1.82 |
| difficult_together | 5deg | 0.437 | 0.248 | 0.316 | 2.33 | |
| difficult_together | 5deg+5m | 0.338 | 0.192 | 0.245 | 2.30 | 1.69 |

EOS Brier 0.127 (P(EOS) 0.22 when should continue, 0.80 when should stop).
Training: best val CE at epoch 7 (1.35), divergence to 2.99 by epoch 60.

**Interpretation.** Precision and range error already beat the evidential
reference (P 0.53 vs 0.45; mADE 1.8 vs 2.3); the entire F1 gap is recall
(0.41 vs 0.71). Two confirmed causes: conservative cardinality (the decoder
emits 81 detections for 105 GT on static_easy) and severe overfitting (the
usable model existed at epoch 7 of 60). Both were predicted by the audit
(brief Section 8, findings 4 and 5).

**Decisions.** (a) attack recall first with length-controlled decoding on the
existing checkpoint, zero training cost (E2); (b) fix the training regime
before further architecture ablations so they compare clean models (E3);
(c) then the v2 component ablations one at a time.

---

## D1: diagnostic battery on the trained v1 checkpoints (no training)

Script: decode-only, both checkpoints, gate sweep, cardinality confusion,
recall by dominance rank, per-token-type val CE, length-controlled arm.
Raw output: notebooks/results/diag_v1.json.

**Symptom A: recall 0.41 vs reference 0.71.** Hypotheses and verdicts:

| hypothesis | discriminating evidence | verdict |
| --- | --- | --- |
| A1 cardinality prior, early EOS | confusion matrix: the decoder NEVER emits a 2nd object on any split (pred N <= 1 in 100% of scenes; 2-GT scenes get 1 or 0 predictions) | CONFIRMED, strong form |
| A2 localization just outside the 5 deg gate | recall 0.41 -> 0.61 when the gate widens 5 -> 10 deg (static_easy); +0.15 on difficult splits | CONFIRMED, second factor |
| A3 masked/2nd-rank objects under-recovered | recall by dominance rank: rank0 0.27-0.43 vs rank1+ 0.16-0.20 | CONFIRMED (relevant to the core hypothesis later) |
| A4 token-CE is a bad model-selection proxy | last.pt (epoch 60) F1 0.602 static_easy vs best.pt (epoch 7) 0.462; 0.457 vs 0.316 on difficult_together | CONFIRMED, dominant factor |

**Symptom B: val CE divergence after epoch 7.** Per-type CE roughly doubles
(range 2.60 -> 5.63, azimuth 1.63 -> 3.09, EOS 0.30 -> 1.13) while detection
F1 RISES sharply. So the divergence is an overconfidence/calibration artifact
of teacher-forced CE, not detection overfitting. The earlier "severe
overfitting" reading of E1 was wrong as stated; corrected here.

**Length-controlled arm (best.pt, min 1 object):** F1 0.462 -> 0.533
static_easy, +0.07 to +0.08 everywhere; confirms A1 mechanically. But plain
greedy on last.pt (0.602) beats even that, so checkpoint selection dominates.

**Surviving explanation.** v1's reported numbers were limited by (1) selecting
the checkpoint by val CE, (2) a hard learned cardinality cap at one object
(train prior: 84% of scenes have <= 1 object), (3) angular error mass in the
5-10 deg band.

**Decisions.** D2 (no training): score last.pt fully (both gates, mADE, EOS,
cardinality confusion, with and without min-1 decode) for the corrected v1
baseline. E3 (training): retrain identically but select checkpoints by val F1
(decoded), 120 epochs, since F1 was still climbing at 60. Predicted effect:
E3 best >= last.pt everywhere; if val F1 plateaus then falls, true detection
overfitting exists after all and we will see where.

## E3: identical architecture, checkpoint selection by decoded val F1 + early stopping

Run `soundreg_e3`. Single change vs E1/v1: model selection. Per-epoch greedy
decode of the 280 val scenes, best checkpoint by val F1@5deg, early stop with
patience 25. Best at epoch 12 (val F1 0.565), stopped at 37. Curves (with the
val-F1 panel): notebooks/runs/soundreg_e3/curves.png.

| split | gate | P | R | F1 | mAAE | mADE |
| --- | --- | --- | --- | --- | --- | --- |
| static_easy | 5deg | 0.581 | 0.648 | **0.613** | 1.98 | |
| static_easy | 5deg+5m | 0.479 | 0.533 | **0.505** | 1.95 | 1.87 |
| static_difficult | 5deg | 0.478 | 0.361 | 0.411 | 1.83 | |
| static_difficult | 5deg+5m | 0.384 | 0.290 | 0.330 | 1.91 | 1.74 |
| easy_together | 5deg | 0.529 | 0.565 | 0.547 | 1.87 | |
| easy_together | 5deg+5m | 0.441 | 0.471 | 0.456 | 1.81 | 1.93 |
| difficult_together | 5deg | 0.465 | 0.348 | 0.398 | 2.02 | |
| difficult_together | 5deg+5m | 0.369 | 0.276 | 0.316 | 1.99 | 1.72 |

EOS Brier 0.137 (continue 0.140, stop 0.726).

**Outcome vs prediction.** Predicted E3 >= v1-last everywhere: confirmed.
static_easy now beats the evidential reference on BOTH gates (0.613 vs 0.5515
and 0.505 vs 0.4706) at one fifth of the v1 training time. mAAE dropped from
~2.4 to ~1.9 across the board. EOS over-stopping pressure eased (P(EOS) when
it should continue: 0.22 -> 0.14).

**Scoreboard vs the 0.65 goal (5 deg gate).** static_easy 0.613 (gap 0.04),
easy_together 0.547 (gap 0.10), static_difficult 0.411 and
difficult_together 0.398 (gap 0.25). The difficult splits are the
battleground; D3 (running) stratifies their misses by range, azimuth sector,
dominance rank, and cardinality to decide the next intervention.

## D3: error stratification on the E3 checkpoint (no training)

Raw: notebooks/results/diag_e3.json (cardinality confusion, recall by range
bin / azimuth sector / dominance rank, gates 5 and 10 deg, all splits).

Findings:
1. The hard one-object cap is GONE with F1-based selection: E3 emits 2 (and
   occasionally 3) objects; on static_difficult, 13 of 54 two-GT scenes now
   get two predictions. Still under-emits on most 2-GT scenes.
2. NEAR-FIELD is the worst range bin everywhere: recall@5deg at 0-10 m is
   0.41 / 0.16 / 0.30 / 0.19 across the four splits, vs 0.84 / 0.42 / 0.81 /
   0.46 at 10-20 m. Suspicion: at close range the angular extent of a car is
   huge (1 m offset at 5 m is ~11 deg), so the annotation center and the
   acoustic centroid (engine, tires) can disagree by more than the 5 deg
   gate; micro-diagnostic will check whether near-field misses are emitted
   but 5-10 deg off (label/physics mismatch) or not emitted at all.
3. Side sector (60-120 deg off axis) is weakest but carries few GT objects
   (7-22 per split); traffic is mostly front/rear. Data scarcity is the
   simpler explanation; no rotation augmentation can fix it (rig physics).
4. Masked sources still trail: rank1+ recall 0.26-0.29 vs rank0 0.38-0.39 on
   difficult splits. Conditioning headroom remains.
5. 0.12-0.16 recall everywhere sits between the 5 and 10 deg gates; with
   mAAE ~1.9 over 5 deg bins (quantization floor 1.25), finer azimuth
   vocabulary should convert part of that band.

**Decisions.** E4: scene-mixing augmentation (STFT superposition of two train
scenes, gain U(-10, 0) dB on the second, union of labels reordered by
gain-adjusted own-scene dominance, 50% of train samples per epoch). Attacks
cardinality prior, masked-source exposure, and the difficult splits at once;
physically exact for the fixed rooftop array. Predicted: rank1+ and 2-GT
recall up, cardinality confusion shifts right, modest precision cost on
sparse scenes. E5: azimuth vocab 72 -> 240, single change vs E3. Predicted:
mAAE 1.9 -> ~1.3, recall@5deg +0.03 to +0.08 from the 5-10 deg band.

## D4: near-field micro-diagnostic (E3 checkpoint)

Recall by range bin across gates (notebooks/results/diag_e3_nearfield.json):
0-10 m recall roughly DOUBLES from gate 5 to gate 15 on every split
(e.g. static_easy 0.41 -> 0.68, difficult_together 0.19 -> 0.53) while
10-20 m saturates near 1.0 by gate 10. Verdict: near-field misses are
emitted but 5-15 deg off, consistent with the annotation-center vs
acoustic-centroid mismatch (1 m offset at 5 m is ~11 deg). Partly a label
geometry effect, not purely model error. Far misses (20-30 m) do NOT recover
with wider gates: those are true absences (low SNR at range).

## E4: scene-mixing augmentation (single change vs E3)

Run `soundreg_e4_mix`. With prob 0.5 a train sample becomes
stft_a + g * stft_b (g uniform in -10..0 dB), labels = union reordered by
gain-adjusted own-scene dominance, capped at MAX_OBJECTS. Best epoch 51,
early stop 76 (augmentation regularizes: peak comes 4x later than E3).

| split | P | R | F1@5deg | mAAE | F1 vs E3 |
| --- | --- | --- | --- | --- | --- |
| static_easy | 0.605 | 0.714 | **0.655** | 1.75 | +0.042 |
| static_difficult | 0.535 | 0.465 | 0.497 | 1.86 | +0.086 |
| easy_together | 0.491 | 0.628 | 0.551 | 1.76 | +0.004 |
| difficult_together | 0.498 | 0.448 | 0.472 | 1.98 | +0.074 |

**Outcome vs prediction.** Confirmed and stronger: recall up everywhere
(static_easy recall 0.714 now MATCHES the evidential reference), the 0.65
goal is reached on static_easy. Physically valid superposition works.

## E5: azimuth vocabulary 240 (single change vs E3)

Run `soundreg_e5_az240`. Best epoch 32, early stop 57.

| split | P | R | F1@5deg | mAAE | F1 vs E3 |
| --- | --- | --- | --- | --- | --- |
| static_easy | 0.450 | 0.552 | 0.496 | 1.63 | -0.117 |
| static_difficult | 0.620 | 0.481 | **0.541** | 2.06 | +0.130 |
| easy_together | 0.473 | 0.543 | 0.505 | 1.78 | -0.042 |
| difficult_together | 0.595 | 0.464 | **0.521** | 2.11 | +0.123 |

**Outcome vs prediction.** Partially contradicted: predicted a uniform gain
from finer bins; got a SPLIT pattern: large gains with much higher precision
on the difficult splits, losses on the easy ones. Possible mechanisms (to be
separated later if it matters): harder 240-way azimuth task changes what the
single val-F1 number selects; precision/recall balance shifts toward
precision. Single-seed caveat applies to all of tonight's runs.

**Decision.** E4 and E5 improve complementary splits; E6 = mixing + az240
combined, same recipe otherwise. Predicted: at or near best-of-both per
split; if the easy-split loss of E5 persists under mixing, the az240 change
is implicated as a real trade-off rather than selection noise.

## E6: mixing + az240 combined

Run `soundreg_e6_mix_az240`. Best epoch 25, early stop 50.

| split | P | R | F1@5deg | F1@5deg+5m | vs best single |
| --- | --- | --- | --- | --- | --- |
| static_easy | 0.600 | 0.686 | 0.640 | **0.533** | E4 0.655 |
| static_difficult | 0.544 | 0.437 | 0.485 | 0.388 | E5 0.541 |
| easy_together | 0.509 | 0.650 | **0.571** | 0.488 | E4 0.551 |
| difficult_together | 0.500 | 0.412 | 0.452 | 0.355 | E5 0.521 |

Cross-config scoreboard @5deg (mean): E3 0.492, E5 0.516, E6 0.537,
E4 0.544. At the joint gate E6's static_easy 0.533 is the best so far
(reference 0.4706).

**Outcome vs prediction.** Partial composition only: E6 wins easy_together,
holds near E4 on static_easy, but loses E5's difficult-split gains.

**Meta-finding (important).** E4, E5, E6 all peak at val F1 ~0.545 yet
differ by up to 0.09 F1 per test split: the single val-F1 selection signal is
blind to the difficult-split axis of the goal. The selection criterion needs
to represent difficult conditions (candidate: also decode a MIXED version of
val and select on the mean), otherwise every future run is selected for easy
performance.

**Decisions.** D5 (running): FiLM contribution, min-1 decode, rank recall on
E6. Next training: attention pooling over T (v2 fix 1, untested) on the E6
recipe, plus mixed-val selection; deeper mixing gains (-20..0 dB) as a
follow-up lever for low-SNR exposure. Single-seed caveat noted for close
calls (E4 vs E6 ranking); the structural gaps (0.15 to goal on difficult)
exceed plausible seed noise.

## D5: decode arms, FiLM contribution, rank recall on E6 (no training)

Raw: notebooks/results/diag_e6.json.

| question | evidence | verdict |
| --- | --- | --- |
| does min-1 decoding still help? | plain vs min1 within 0.01 F1 on every split | obsolete; mixing fixed EOS conservatism at the root |
| is FiLM contributing? | vel0 arm: -0.047 F1 on easy_together (motion split), ~0 elsewhere (static splits have v=0 anyway) | FiLM stays; replacement hypothesis rejected |
| did mixing fix masked-source recall? | rank1+ vs rank0 recall: 0.667 vs 0.649 on easy_together (parity); 0.36-0.38 vs 0.43-0.46 on difficult (gap halved vs D3) | yes; the conditioning mechanism demonstrably recovers masked sources |

Cardinality: 2-GT scenes now receive 2 predictions in ~30% of cases
(was ~0% in v1).

**Decisions.** E7 = E6 + attention pooling over T (the untested v2 fix 1;
content-adaptive weights over the 19 frames instead of the mean). E8 = E6
with mixing gain widened to -20..0 dB (deeper maskers = more low-SNR
exposure, aimed at the far-range true absences). Then combine winners with
mixed-val selection (the D-meta fix for the blind selection criterion).

## E7: attention pooling over T (single change vs E6)

Run `soundreg_e7_attnpool`. Content-adaptive softmax weights over the 19
frames replace the mean. Best epoch 28, stop 53.
F1@5deg: 0.618 / 0.468 / 0.557 / 0.474 (mean 0.529 vs E6 0.537).

**Outcome vs prediction.** NOT confirmed. Predicted a clear gain from
preserving per-frame masking structure; got a wash (slightly better
difficult_together, slightly worse elsewhere, and mAAE worsened ~0.2 deg).
Honest reading: with 100 ms windows and mostly stationary traffic noise, the
frames may carry too little differential information for pooling to matter,
or the single-channel scoring is too weak an attention. The v2 figure's
fix 1 should be downgraded from "expected win" to "tested, neutral".

## E8: deeper mixing gains, -20..0 dB (single change vs E6)

Run `soundreg_e8_deepmix`. Best epoch 19, stop 44.
F1@5deg: 0.654 / 0.465 / 0.575 / 0.477 (mean 0.543, statistically tied with
E4's 0.544; best easy_together so far).

**Outcome vs prediction.** Mild confirmation: difficult_together +0.025 and
easy_together +0.004 vs E6, static_difficult -0.020. Deeper maskers help the
together splits slightly; no breakthrough on far/low-SNR absences.

**Standing pattern across E4-E8.** Mixing configs cluster at mean 0.54 with
recall-heavy operating points; E5 (no mixing, az240) still owns the
difficult splits via a precision-heavy operating point (P ~0.6). The
per-object confidence is exported but unused: D6 (running) sweeps a
confidence threshold tuned on val / mixed-val per the benchmark protocol,
which should let one config reach both operating points.

## D6: confidence-threshold sweep (no training): NEGATIVE

Thresholds tuned on val and on a mixed copy of val, applied to test
(notebooks/results/diag_confsweep.json). Effect on E6: mean F1 0.537 ->
0.540. On E8: the plain-tuned threshold HURTS (0.543 -> 0.533); the
mix-tuned sweep selects no threshold at all. Verdict: the per-object
confidence (mean token probability) is not informative enough to trade
recall for precision; E5's difficult-split advantage lies in WHICH
detections it makes, not in a filterable low-confidence subset.

## E9: mixed-val checkpoint selection: NEGATIVE

Run `soundreg_e9_mixsel` (E8 recipe, selection = mean of plain-val and
mixed-val F1). Mean F1 0.496, worse on every split than E8. Verdict:
synthetic mixtures are a poor proxy for the difficult splits; their
difficulty comes from hard positions and low SNR, not from object count.
Plain val-F1 selection stays.
