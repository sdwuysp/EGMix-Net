# EGMix-Net: Energy-Gated Kernel Mixing and Spectral Bias-Aware Training for Lightweight Image Dehazing

[![PyTorch](https://img.shields.io/badge/PyTorch-1.9.0+-red.svg)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-green.svg)](https://opensource.org/licenses/MIT)

This repository contains the official PyTorch implementation of the paper **EGMix-Net: Energy-gated kernel mixing and spectral bias-aware training for lightweight image dehazing**.

## 🚀 Model Zoo & Pretrained Weights

We provide the pretrained models for all datasets and scaling variants evaluated in the paper. 

**[🔗 Download All Pretrained Weights (Google Drive)](https://drive.google.com/drive/folders/1VGv1NUkKZ-5N8LPxbOJ3lfE4eLhn_6XY)**

| Model Variant | Dataset | PSNR (dB) | SSIM | Weights |
| :--- | :---: | :---: | :---: | :--- |
| **EGMix-Net (S)** | SOTS-Indoor | 40.41 | 0.995 | `egmix_net_its.pth` |
| **EGMix-Net (S)** | SOTS-Outdoor | 37.09 | 0.993 | `egmix_net_ots.pth` |
| **EGMix-Net (S)** | NH-HAZE | 19.80 | 0.683 | `egmix_net_nhhaze.pth` |
| **EGMix-Net (S)** | Dense-HAZE | 16.07 | 0.602 | `egmix_net_densehaze.pth` |
| **EGMix-Net-T** | SOTS-Indoor | 37.32 | 0.991 | `egmix_net_t_its.pth` |
| **EGMix-Net-B** | SOTS-Indoor | 40.72 | 0.996 | `egmix_net_b_its.pth` |

*Note: The outdoor results are evaluated on the official RESIDE SOTS-Outdoor release. Please refer to our paper for the evaluation discrepancy details.*

## ⚙️ Installation

```bash
git clone https://github.com/[Your-Username]/EGMix-Net.git
cd EGMix-Net
pip install -r requirements.txt
```

## 📂 Data Preparation

Please organize the datasets (RESIDE, NH-HAZE, Dense-HAZE) following this structure:

```text
datasets/
├── SOTS/
│   ├── indoor/
│   │   ├── hazy/
│   │   └── clear/
│   └── outdoor/
├── NH-HAZE/
└── Dense-HAZE/
```
*Modify the data paths in the corresponding JSON files under `configs/`.*

## 🏃 Training and Evaluation

### Evaluation
To test a pretrained model on a specific dataset, run:
```bash
python test.py --config configs/indoor.json --checkpoint path/to/egmix_net_its.pth
```

### Training
To train EGMix-Net from scratch (or with the LF consistency loss schedule), simply run:
```bash
python train.py --config configs/indoor.json
```
*The training configurations, including the $\mathcal{L}_{\mathrm{LF}}$ schedule and the $5\times5$ parallel energy-gated convolution settings, are fully specified in the `configs/*.json` files.*

## 📏 Diagnostics Tools
We provide a set of diagnostic scripts under `tools/` for reproducibility:
* `python tools/eval_dir.py` - Standard metrics evaluation (PSNR/SSIM).
* `python tools/count_macs.py` - MACs and Parameters computation.
* `python tools/profile_model.py` - Latency and throughput profiling.
* `python tools/reparam.py` - Structural re-parameterization to fuse parallel branches for fast inference. 
* `python tools/bench_speed.py` - Latency and throughput profiling (e.g., 1024x1024 FP32 on RTX 3090).

## 📜 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.