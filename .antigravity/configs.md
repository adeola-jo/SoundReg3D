# AutoReg3D Training and Hyperparameter Configurations

This document outlines the default configurations for Supervised Fine-Tuning (SFT) and Reinforcement Learning (RL) training inside [autoreg3d/tools/cfgs/autoreg3d_models/](file:///home/joadeola/Softwares/SoundReg3D/autoreg3d/tools/cfgs/autoreg3d_models).

---

## 1. Supervised Fine-Tuning (SFT)
* **Configuration File**: [autoreg3d_conv_voxel.yaml](file:///home/joadeola/Softwares/SoundReg3D/autoreg3d/tools/cfgs/autoreg3d_models/autoreg3d_conv_voxel.yaml)
* **Model Class**: `AutoReg3D`
* **Dataset**: nuScenes (discretized to $360 \times 60$ BEV grid)
* **Target Tokenizer Config**:
  * Range limits: $x \in [0, 30]$, $y \in [0, 10]$, $z \in [0, 10]$, yaw $\in [-\pi, \pi]$, velocity $v_x, v_y \in [-30, 30]$.
  * Bin sizes (quantization steps): XYZLWHT = `[2160, 2160, 160, 600, 200, 200, 125, 600, 600]`.
  * Class vocabulary: 11 tokens (10 classes + noise class).
  * Class token placement: `first`.
  * Maximum sequence length: 1002 tokens.
* **Optimization**:
  * Batch Size: 16 per GPU.
  * Total Epochs: 20.
  * Optimizer: `adamW_onecycle`.
  * Learning Rate: 0.001 (using cosine warm-up / one cycle scheduler).
  * Weight Decay: 0.01.
  * Backbone Schedule: Frozen for the first 10 epochs (`UNFREEZE_BACKBONE_EPOCH: 10`), then unfrozen and trained jointly with the decoder.

---

## 2. Reinforcement Learning (RL) via GRPO
* **Configuration File**: [autoreg3d_conv_voxel_rl.yaml](file:///home/joadeola/Softwares/SoundReg3D/autoreg3d/tools/cfgs/autoreg3d_models/autoreg3d_conv_voxel_rl.yaml)
* **Model Class**: `AutoReg3DRL`
* **Policy Optimization Parameters (`RL_CONFIG`)**:
  * Epsilon Clipping: 0.2 (`EPSILON: 0.2`).
  * Group Size (generations per scene): 8 (`NUM_GENERATION_IN_GROUP: 8`).
  * Temperature: 1.0.
  * Importance Sampling level: `token`.
  * KL Reference Penalty (beta): 0.0.
  * Velocity Masking: `IGNORE_VELOCITY_IN_RL_LOSS: True` (avoids punishing velocity during RL exploration).
  * Decoding Mode during rollouts: `sample`.
* **Optimization**:
  * Batch Size: 16 per GPU.
  * Total Epochs: 1.
  * Optimizer: `adamW` (flat learning rate).
  * Learning Rate: 0.0001.
  * Backbone Schedule: Frozen throughout training (`UNFREEZE_BACKBONE_EPOCH: -1`).
