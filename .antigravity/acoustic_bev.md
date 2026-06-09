# AcousticBEV: Evidential (K-Slot MoG) Subproject Documentation

This document summarizes the architectural details, datasets, loss functions, and designs found inside the **Acoustic-BEV K-slot evidential project** located at [Acoustic-BEV/src/evidential/](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential).

---

## 1. Overview & Paradigm Shift

Unlike standard AcousticBEV, which predicts dense $360^\circ \times 60$ grids using polar heatmaps, the subproject in `src/evidential/` represents sound sources as **discrete objects** associated with **uncertainty coordinates**.
* Bypasses spatial grids entirely.
* Implements a **K-slot Mixture-of-Gaussians (MoG) slot-based architecture** (conceptually inspired by DETR object queries).
* Solves object assignment using **Hungarian matching** between predictions and raw ground-truth coordinates.
* Directly outputs spatial coordinates alongside calculated angular and range variances.

---

## 2. Core Components

### 2.1 Dataset Loader ([dataset.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/data/dataset.py))
* **Inputs**:
  * `stft`: Multi-channel audio STFT spectrogram of shape `(64, T=19, F=84)`. Constructed by splitting 32 complex channels into real/imaginary parts.
  * `velocity`: linear speed magnitude scalar `(1,)` representing the ego-vehicle velocity.
* **Targets**: Continuous target list of shape `(N_obj, 2)`, where each target is `[theta_deg, r_m]`.
* **No Grid targets**: Heatmaps and grid index conversions are completely omitted.

### 2.2 Neural Network & Head Architectures

The model structure is implemented in [network.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/models/network.py). It encodes velocity with `SpeedFiLM2Ch_Factorized`, runs a convolutional ResNet trunk, and passes the output to a pluggable slot head:

* **Slot Heads** ([heads.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/models/heads.py)):
  1. **`attention` (Default)**: DETR-style learned queries ($K \times C$) attend over the spatial-angular features using Multi-Head Cross-Attention.
  2. **`global_mlp`**: Compresses all spatial features via a pooling layer and projects them to $K$ slot queries using a Multi-Layer Perceptron.
  3. **`sector`**: Assigns slots to fixed spatial sectors of the bird's-eye view.
* **Slot Output Contract**:
  For each slot $k$ and mixture component $m$:
  * $q_k$: Presence probability (sigmoid activated).
  * $\pi_{k,m}$: Mixture weight (softmax activated over components).
  * $\mu_{\theta,k,m}$, $\mu_{r,k,m}$: Angular/range point estimates (range uses softplus activation).
  * $\sigma^2_{\theta,k,m}$, $\sigma^2_{r,k,m}$: Estimated variances (softplus activated with a small epsilon floor).

---

## 3. Loss Mechanics & Hungarian Matching ([losses.py](file:///home/joadeola/Softwares/Acoustic-BEV/src/evidential/nn/losses.py))

### 3.1 Hungarian Matching
To assign the ground-truth targets to predicted slots during training, a cost matrix $C$ is computed under `torch.no_grad()`:
$$C[n, k] = -\log p_k(\theta_n, r_n) - \log q_k$$
A linear sum assignment (Hungarian matching) assigns ground-truth objects $n$ to unique slots $k$.

### 3.2 Loss Formulation
The total loss is a unified likelihood representation split into two parts:
$$L = L_{\text{flag}} + L_{\text{cloud}}$$

1. **Flag Loss ($L_{\text{flag}}$)**:
   * Binary Cross Entropy (BCE) over all $K$ slots. Matched slots are supervised to $1.0$, unmatched slots to $0.0$.
2. **Cloud Loss ($L_{\text{cloud}}$)**:
   * Negative Log-Likelihood (NLL) of the Mixture of Gaussians evaluated only for the matched pairs:
     $$p(\theta, r) = \sum_m \pi_m \cdot \mathcal{N}_{\text{circ}}(\theta; \mu_{\theta,m}, \sigma^2_{\theta,m}) \cdot \mathcal{N}(r; \mu_{r,m}, \sigma^2_{r,m})$$
   * Uses a **circular Gaussian** implementation for angles to correctly handle the $0/360^\circ$ wrap-around difference:
     $$\Delta_\theta = ((\theta_1 - \theta_2 + 180^\circ) \bmod 360^\circ) - 180^\circ$$
