# Repository Exploration Notes: Acoustic-BEV and SoundReg3D

This document summarizes the structures, architectures, and objectives of the **Acoustic-BEV** (including the deep-dive into the `src/evidential` subdirectory) and **SoundReg3D** repositories, and analyzes the user's ultimate goal in combining or extending these systems.

---

## 1. Executive Summary
- **Acoustic-BEV (Base/Legacy)**: An acoustic bird's-eye-view (BEV) object detection system. It processes multi-channel audio STFT features and ego-vehicle velocity using ResNet backbones conditioned via SpeedFiLM. Traditionally, it outputs a continuous `(60, 360)` polar grid representing spatial sound source intensity heatmaps, optimized via MSE loss.
- **Acoustic-BEV (Active Evidential/Slot Subproject in `src/evidential`)**: A standalone, self-contained project that shifts away from dense grids and Gaussian heatmaps entirely. Instead, it implements a **K-slot Mixture-of-Gaussians (MoG) slot-based architecture** (similar to DETR). It uses K learned query vectors in a cross-attention slot head to attend to angular features. Each slot predicts a presence probability ($q$), mixture weights ($\pi$), polar coordinate estimates ($\mu_\theta, \mu_r$), and uncertainty/variances ($\sigma^2_\theta, \sigma^2_r$). Targets are matched using Hungarian matching. The scientific focus is **uncertainty calibration and reliability** of slot-based acoustic perception.
- **SoundReg3D / AutoReg3D**: A LiDAR-based 3D object detector reformulated as an **autoregressive sequence modeling** task. Instead of using a dense, hand-crafted regression head (like CenterPoint), it tokenizes 3D bounding boxes into a sequence of discrete tokens (classes + coordinates + dimensions + heading) and uses a GPT-2-based Transformer Decoder cross-attending to 2D BEV features to generate boxes one token at a time.
- **The Convergence (SoundReg3D)**: The name **SoundReg3D** implies adapting the autoregressive bounding-box sequence modeling approach to acoustic BEV features. Instead of relying on a dense (60, 360) polar grid or a Hungarian slot-matching model, the user plans to use the **Acoustic-BEV feature extraction pipeline** (multi-channel audio STFT + ego speed conditioning) to construct BEV feature maps, and then feed them directly into the **AutoReg3D sequence decoder** to regress 3D bounding boxes autoregressively.

---

## 2. Exploration of `Acoustic-BEV/src/evidential` (K-Slot MoG)

The subproject in `src/evidential` is a self-contained workspace managed by `pixi` and implemented in PyTorch Lightning. It bypasses dense polar grids to model sound sources as discrete objects with uncertainty estimates.

### 2.1 Core Data Contract & Target Representation
- File: [dataset.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/data/dataset.py)
- **Input Channels**: 32-channel audio mapped to STFT features of shape `(64, T=19, F=84)`.
- **Velocity**: Extracted as a 2D linear speed magnitude scalar of shape `(1,)`.
- **No Grid/Heatmap GT**: All `P` and `W` heatmap targets are removed from this path. Instead, targets are formatted as a continuous list of variables `[theta_deg, r_m]` of shape `(N_obj, 2)`.

### 2.2 Network Architecture (`MoGSlotNetwork`)
- File: [network.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/models/network.py)
- **Conditioning**: `SpeedFiLM2Ch_Factorized` conditions microphone channels, time, and frequencies based on the ego-vehicle's velocity.
- **Conv Trunk**: ResNet-style convolutional trunk projects the features down to `(B, 100, T, n_theta)`.
- **Pluggable Slot Heads** ([heads.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/models/heads.py)):
  - **`attention` (Default)**: DETR-style learned queries ($K \times C$) attend over the angular spatial tokens via cross-attention.
  - **`global_mlp`**: Pools features to a global summary vector, projecting them to $K$ slots.
  - **`sector`**: Fixed angular-sector baseline.
- **Slot Outputs**:
  - `q`: Slot confidence / presence probability (activated via `sigmoid`).
  - `pi`: Mixture weights over $M$ components (activated via `softmax`).
  - `mu_theta` & `mu_r`: Angular/range point estimates (range activated via `softplus`).
  - `sigma2_theta` & `sigma2_r`: Angular/range predicted variances (activated via `softplus + EPS`).

### 2.3 Hungarian Matching and Mixture of Gaussians Loss
- File: [losses.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/losses.py)
- **Cost Matrix for Matching**:
  $$C[n, k] = -\log p_k(\theta_n, r_n) - \log q_k$$
  A Hungarian linear sum assignment assigns ground-truth objects to slots under `no_grad()`.
- **Flag Loss ($L_{\text{flag}}$)**: BCE over all $K$ slots. Matched slots are supervised toward $1.0$, unmatched slots toward $0.0$.
- **Cloud Loss ($L_{\text{cloud}}$)**: Negative Log-Likelihood of the Mixture of Gaussians ($M$ components) evaluated for the matched slot-target pairs. Circular angular differences are handled via:
  $$\Delta_\theta(a, b) = ((a - b + 180^\circ) \bmod 360^\circ) - 180^\circ$$
- **Total Loss**:
  $$L = L_{\text{flag}} + L_{\text{cloud}}$$
  No `lam_loc` or loss balancing hyperparameter is needed, as both terms derive from a joint likelihood.

### 2.4 Active Research Focus & Diagnostics
As detailed in [novelty_plan.md](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/docs/novelty_plan.md):
- **Matched-slot uncertainty calibration**: Developing metrics (like coverage at 1/2/3 $\sigma$, $Z^2$-score calibration, and binned calibration plots) to ensure predicted slot variances track empirical errors.
- **Angle-range uncertainty asymmetry**: Analyzing why angular uncertainty and range uncertainty behave differently under the same probabilistic objectives (angle uncertainty is physically and mathematically harder to calibrate on a flat circular topology).
- **Observability Maps**: Mapping uncertainty and error over physical BEV coordinates to reveal overconfident or weak regions (linked to microphone geometry).
- **Failure Taxonomy**: Tracking model bugs: *overconfident wrong*, *missed active*, *false active*, *bad range*, and *bad angle*.
- **Risk-controlled localization**: Using uncertainty ($\Phi = 2\nu + \alpha$ or variance thresholds) to filter/reject risky predictions at inference.

---

## 3. Exploration of `SoundReg3D`

### 3.1 Codebase Structure
- `autoreg3d/`: Contains the main AutoReg3D codebase, built on OpenPCDet.
- `autoreg3d_page/`: Contains the project website files hosted at `tzmhuang.github.io/autoreg3d`.

### 3.2 AutoReg3D Core Concepts
1. **Sequence Tokenizer** ([sequence_utils.py](file:///home/joadeola/Softwares/SoundReg3D/autoreg3d/pcdet/utils/sequence_utils.py)):
   - Quantizes continuous bounding box parameters (center coordinates, sizes, headings, and optionally object velocity) and class IDs into a single 1D sequence of discrete tokens.
   - Vocab layout: `[BOS] [class] [x, y, z] [dx, dy, dz] [heading] [vx, vy] [EOS]` (interleaved per config).
2. **GPT-2 Decoder** ([autoreg3d_head.py](file:///home/joadeola/Softwares/SoundReg3D/autoreg3d/pcdet/models/dense_heads/autoreg3d_head.py)):
   - Loads a standard Hugging Face GPT-2 transformer decoder (with cross-attention enabled).
   - The encoder's 2D BEV features (from a 3D backbone like PointPillars, Voxels, or DSVT) are projected and passed as `encoder_hidden_states` (keys/values) to the decoder.
   - The decoder takes the tokenized bounding boxes as input (`input_ids`) and predicts the next token in the sequence.
3. **Training & Inference Modes**:
   - **SFT (Supervised Fine-Tuning)**: Standard teacher forcing.
   - **RL (Reinforcement Learning)**: Fine-tuning via GRPO (Group Relative Policy Optimization) on task-aligned metrics (e.g. F1-score/precision/recall).
   - **Evaluation**: Employs greedy sampling or cascading refinement (predicting boxes from partial box sequences).

---

## 4. The User's Ultimate Goal & Future Directions

The integration of **Acoustic-BEV** and **AutoReg3D** represents a novel shift in 3D object detection from sound.

```mermaid
graph TD
    subgraph Acoustic Feature Extractor (Acoustic-BEV)
        A[32-Ch Audio + Ego Speed] --> B[STFT Spectrograms]
        B --> C[SpeedFiLM Conditioning]
        C --> D[ResNet Backbone Trunk]
        D --> E[Spatial BEV Features 2D]
    end

    subgraph Autoregressive Sequence Decoder (AutoReg3D)
        E --> F[Transformer Decoder Cross-Attention]
        G[Box Token Sequence input_ids] --> H[GPT-2 Decoder Model]
        F --> H
        H --> I[Next-token prediction / Autoregressive boxes]
    end
```

### Potential Avenues of Integration:
1. **Acoustic Autoregressive Detector**: Replace the Point Cloud / LiDAR 3D backbones in AutoReg3D with the STFT + ResNet backbone from Acoustic-BEV. Instead of regressing heatmaps or matching slots, decode the 3D bounding boxes (x, y, z, dimensions, yaw) directly and autoregressively.
2. **Multi-Modal Fusion**: Construct a joint BEV feature representation using both LiDAR features (PointPillars/Voxels) and Acoustic features (from the STFT ResNet). Pass this joint BEV map to the GPT-2 decoder to autoregressively decode sequence boxes, allowing sound to guide detection in scenarios with occlusions or poor visibility (e.g. night, fog, behind walls).
3. **Evidential / Probabilistic Slot Sequence Modeling**: Use the predicted slot confidences ($q$) and uncertainties ($\sigma^2$) from `Acoustic-BEV/src/evidential/` to guide the token generation process or filter bounding boxes generated by the AutoReg3D decoder.
