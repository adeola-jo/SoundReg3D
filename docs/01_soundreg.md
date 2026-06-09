# SoundReg on Acoustic-BEV: documentation for `notebooks/01_soundreg.ipynb`

SoundReg casts acoustic object detection as autoregressive sequence
generation. From one multichannel STFT window of the car's microphone
array, the model emits the set of sounding objects, one object at a time,
loudest first, each conditioned on the objects already emitted. This
document explains the data, the geometry conventions, the math, the model,
and how to run and read the notebook. The working brief
(`soundreg_brief (1).pdf`) is the source of the formulation; AutoReg3D
(arXiv:2603.07985) is the source of the sequence-detection recipe.


# =====================================================================
# 1. Problem and idea
# =====================================================================

From a short window A of the 32-microphone array (audio only at
inference), output the set of road users currently making sound, each as
(range, azimuth) in the BEV plane:

    S = {(r_i, theta_i)},   i = 1..N,   N unknown and variable.

Masking is the acoustic analog of occlusion: a loud source hides weaker
concurrent sources in the time-frequency bins it dominates. SoundReg
therefore generates objects in dominance order (loudest first) and
conditions each prediction on those already emitted. The ordering is an
inductive bias intended to help the model recover low-SNR masked sources.

The headline prediction, and its falsifier: SoundReg's recall advantage
over parallel detectors should GROW as per-source SNR drops and overlap
rises. A flat advantage falsifies the masking idea. The known opposite
failure is the decoder firing `[EOS]` early and silently dropping quiet
sources; the notebook tracks EOS calibration for exactly this reason.


# =====================================================================
# 2. Data
# =====================================================================

The notebook trains on the same `nn_data` pickles the
acoustic-bev-benchmark uses. Default location
`/home/joadeola/Datasets/nn_data`, overridable with the `SOUNDREG_DATA`
environment variable.

| Split file | Role |
| --- | --- |
| `train.pkl`, `val.pkl` | training and model selection |
| `test_static_easy.pkl` | single object, no ego motion |
| `test_static_difficult.pkl` | single object, hard positions |
| `test_easy_together.pkl` | multiple objects (up to 3) |
| `test_difficult_together.pkl` | multiple objects, hard conditions |

Pickle contract (identical to the evidential dataset):

    {
      "order": [token, ...],            # sample ids, hex strings
      "items": {
        token: {
          "stft":     ndarray (64, F=84, T),   # see channel layout below
          "ann":      [{"center": [x, y, z]}, ...],  # sounding objects only
          "velocity": {"lin_x": float, "lin_y": float},
        }, ...
      }
    }

STFT details. 48 kHz audio, n_fft 512, hop 256. The 84 frequency bins span
[100, 8000] Hz. T varies per clip and is padded or truncated to 19 frames.
The 64 channels are interleaved per microphone: mic i contributes real
parts on channel `2i` and imaginary parts on channel `2i+1` (confirmed in
`cut_annotation.py` and `mic_geometry.yaml`). The model consumes
`(B, 64, T=19, F=84)`; the dominance beamformer reconstructs the complex
STFT as `stft[0::2] + 1j * stft[1::2]`, shape `(32, 84, 19)`.

Targets. Each annotation center `[x, y, z]` becomes

    theta_deg = degrees(atan2(y, x)) % 360      # in [0, 360)
    r_m       = hypot(x, y)                     # objects beyond 30 m dropped

The annotated boxes are the sounding objects; silent objects are simply
not annotated, so the label set IS the detection target set. A single
class is used, so the class token of the brief drops out and one object is
just the pair (r, theta).

Velocity. The dict reduces to the linear speed scalar
`hypot(lin_x, lin_y)`, shape `(1,)`, consumed by SpeedFiLM.


# =====================================================================
# 3. Geometry and the dominance score
# =====================================================================

Microphone positions come from `metadata/mic_on_the_car.xml`: 32 `<pos>`
elements with x, y, z attributes.

Why the axis swap. The XML stores `(x_forward, y_up, z_lateral)`. The
evidence is in the numbers: y takes only two values across all 32 mics
(-0.022 and -0.102, the two mounting heights of the two arrays), while z
spans about 1.1 m across the car. Annotations, however, define
`theta = atan2(y, x)` in the ground plane, where y is LATERAL. Steering
computes mic-to-source distances, so both must live in one frame; the
loader therefore reorders the XML columns to `(x_fwd, lateral, up)` via
`pos[:, [0, 2, 1]]`. This is the same swap the benchmark's Neural-SRP
notebook applies. Without it the beamformer would read mounting height as
lateral offset and every delay would be wrong.

Steering and dominance. Near-field delay-and-sum steering with spherical
spreading, sources assumed at 1 m height:

    a_m(p, f) = (1 / d_m) * exp(-j 2 pi f d_m / c),   d_m = ||p - p_m||
    w(p, f)   = a(p, f) / ||a(p, f)||

The dominance score of object i (brief Eq. 3) is the steered-response
power of the mixture at the labeled position:

    d_i = sum over t, f of | w(r_i, theta_i)^H X(t, f) |^2

Objects are emitted in descending d_i. Dominance uses raw power, not
PHAT whitening, because it measures energy, not detectability.


# =====================================================================
# 4. Tokenization
# =====================================================================

Vocabulary, one shared embedding space with contiguous blocks:

    [ PAD=0, BOS=1, EOS=2 | range x 24 | azimuth x 72 ]

The polar grid IS the output vocabulary: 24 range bins of 1.25 m over
[0, 30] m, 72 azimuth bins of 5 degrees over [0, 360). Azimuth is finer
than range, and the 5 degree bin width matches the strictest benchmark
gate. Decoding returns bin centers.

A scene in emission order pi is

    [BOS, (t_r, t_theta)_pi(1), ..., (t_r, t_theta)_pi(N), EOS]

`[EOS]` carries the cardinality. There is no slot count K and no
confidence threshold anywhere in the detector. At decode time, token-type
masking keeps sequences structurally valid: range steps may only emit
range tokens, azimuth steps only azimuth tokens, and EOS is legal only at
object boundaries (a half-emitted object is not a thing).


# =====================================================================
# 5. Model
# =====================================================================

The encoder is the MoG-slot network's front end from
`Acoustic-BEV/src/evidential`, re-implemented faithfully. The decoder is
the only new part.

| Stage | Layer | Output shape |
| --- | --- | --- |
| input | stft + velocity | (B, 64, 19, 84) + (B, 1) |
| FiLM | SpeedFiLM2Ch_Factorized (hidden 64) | (B, 64, 19, 84) |
| conv1 | Conv2d(64 -> 128, k=(1,7), s=(1,2)), BN, ReLU | (B, 128, 19, 39) |
| conv2 | Conv2d(128 -> 256, k=(1,5), s=(1,2)), BN, ReLU | (B, 256, 19, 18) |
| res | 2 x ResidualBlock(256) | (B, 256, 19, 18) |
| conv3 | Conv2d(256 -> 360, 1x1), BN, ReLU | (B, 360, 19, 18) |
| permute | (0, 3, 2, 1) | (B, 18, 19, 360) |
| conv4 | LazyConv2d(-> 100, 1x1), BN, ReLU | (B, 100, 19, 360) |
| tokenize | mean over T, transpose | (B, 360, 100) |
| project | Linear(100 -> 192) + azimuth position embedding | (B, 360, 192) |

Implementation notes that matter:

- The convolutions are UNPADDED, so F runs 84 -> 39 -> 18. The original
  uses `LazyConv2d` for conv4 precisely so nobody has to track that
  number; the notebook does the same, with one dummy forward to
  materialize the lazy weights before the optimizer is built.
- SpeedFiLM factorizes gamma and beta over (mic, re/im), time, and
  frequency, sums the three factors, and applies
  `x * (1 + gamma) + beta`, so zero output equals identity conditioning.
- The tokenization (mean over T, angular axis becomes the sequence) is
  exactly what the AttentionSlotHead does before its K queries attend.
  SoundReg keeps that token sequence and changes only who attends to it.

The head: a causal Transformer decoder (4 layers, d_model 192, 6 heads,
FF 768, pre-norm) cross-attends over the 360 angular tokens and predicts
the next token of the object sequence. Token, position, and the tied
output head share one embedding table.

Two pitfalls baked into the implementation, found the hard way:

- `nn.TransformerDecoder` with `norm_first=True` leaves the residual
  stream unnormalized; without an explicit final `LayerNorm` the tied
  logits start at a cross-entropy near 100 instead of ln(V) ~ 4.6.
- `nn.Embedding` defaults to N(0, 1), far too hot for tied embeddings;
  everything is re-initialized at std 0.02.


# =====================================================================
# 6. Training
# =====================================================================

Teacher forcing with one cross-entropy over all token types (brief
Eq. 6). The input is `tokens[:, :-1]`, the target `tokens[:, 1:]`, PAD
positions excluded. There are no per-attribute losses, no matching, and
no loss weights; the single-loss property is half the appeal of the
formulation.

| Knob | Default |
| --- | --- |
| epochs / batch | 60 / 32 |
| optimizer | AdamW, lr 3e-4, weight decay 0.01 |
| schedule | 5% linear warmup, cosine to zero |
| grad clip | 1.0 |
| seed | 42069 (benchmark-wide) |

Per-epoch logs: token cross-entropy and next-token accuracy on train and
val. `history.csv`, `best.pt` (lowest val loss), and `last.pt` land in
`notebooks/runs/soundreg_v1/`.


# =====================================================================
# 7. Inference and evaluation
# =====================================================================

Greedy decoding with token-type masking, capacity capped at 6 objects
(benchmark scenes hold at most 3). Each decoded object carries a
diagnostic confidence, `exp(mean token log-prob)`; it is exported in the
prediction CSVs but never used as a threshold. P(EOS) is recorded at
every object boundary.

Predictions are written per split in the benchmark CSV format
(`pred_soundreg_<split>.csv`, columns `sample_id, model_name, confidence,
theta_deg, range_m, class_name, sigma_theta_deg, sigma_r_m`) and scored
with the benchmark matcher, inlined verbatim: greedy one-to-one matching
within the angle gate (5 degrees), plus the joint range gate (5 m) for
the range-capable row. Metrics per split: precision, recall, F1, mAAE
(matched angle error), mADE (matched range error). The rows in
`notebooks/results/metrics_soundreg.csv` are directly comparable to
`benchmark_table_full.csv`; the evidential MoG-slot reference is F1
0.5515 at 5 degrees on static_easy, 0.4706 at 5 degrees + 5 m.

EOS calibration. Stopping after k emitted objects is correct iff
k >= N_gt. The notebook reports the Brier score of P(EOS) against that
label, plus the mean P(EOS) when the decoder should continue versus when
it should stop. Read this BEFORE interpreting weak recall on the
`*_together` splits: early EOS is the failure mode that looks like
"the idea does not work" but is actually a calibration problem.


# =====================================================================
# 8. Running
# =====================================================================

On the machine that has the data:

```bash
jupyter lab notebooks/01_soundreg.ipynb     # then Run All
```

Elsewhere, point at the data first:

```bash
export SOUNDREG_DATA=/path/to/nn_data
```

Quick end-to-end check: set the smoke knobs in Settings
(`LIMIT_TRAIN = 256`, `LIMIT_VAL = 64`, `LIMIT_TEST = 64`, `EPOCHS = 3`).

Output layout:

    notebooks/
    ├── runs/soundreg_v1/
    │   ├── history.csv            per-epoch train/val loss and accuracy
    │   ├── best.pt                lowest val loss checkpoint
    │   └── last.pt
    └── results/
        ├── pred_soundreg_<split>.csv    benchmark-format predictions
        └── metrics_soundreg.csv         one row per (split, gate)

Both directories are gitignored.


# =====================================================================
# 9. Knobs, ablations, and known gaps
# =====================================================================

Every planned ablation is one Settings change:

| Question | Knob |
| --- | --- |
| Does dominance specifically matter? | `ORDERING = "random"` or `"near_to_far"` |
| Vocabulary resolution | `N_R`, `N_THETA` |
| Capacity | `D_MODEL`, `N_LAYERS`, `EPOCHS` |
| Scene cap | `MAX_OBJECTS` |

Known gaps relative to the brief, deliberate for now:

1. Per-source SNR stratified recall (the headline test). The brief says
   to estimate per-source SNR on real data by beamforming toward each
   labeled source. The dominance score already computes that steered
   power, so this is a small evaluation addition once the first real
   run exists.
2. Beam search and sample-then-cluster decoding (greedy only for now).
3. The brief's hand-built encoder features (log-mag + IPD, GCC-PHAT,
   SRP maps with a full polar positional encoding) as encoder ablations
   against the reused trunk.
4. The conditioning ablation (predict each object without seeing
   previously emitted ones), which isolates conditioning from
   tokenization.
