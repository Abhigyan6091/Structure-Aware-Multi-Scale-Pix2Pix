# Structure-Aware Multi-Scale Pix2Pix
**Attention-Augmented Image-to-Image Translation with Perceptual and Edge Consistency Losses**

## Overview
This project implements an enhanced Pix2Pix architecture for paired image-to-image translation tasks.

**Key improvements include:**
- Multi-scale discriminators (PatchGAN + Global)  
- Attention-augmented U-Net generator  
- Perceptual VGG-based loss  
- Edge-aware Sobel consistency loss  

These additions improve:  
- global structural alignment  
- local textural realism  
- perceptual similarity  
- edge sharpness  

## Tasks
**Primary demonstrated task:**
- Maps → Satellite image translation  

**Other suitable tasks:**
- Sketch → Photo  
- Segmentation → Real image  
- CT → MRI  
- Night → Day  
- Facades → Building  

## Architecture Concept
```
Input → Generator (U-Net + Attention) → Fake Image
               │
               ├→ PatchGAN Discriminator (local texture)
               └→ Global Discriminator (global structure)

Total Loss = Adversarial + L1 + Perceptual + Edge
```

## Novel Contributions vs Original Pix2Pix
- Addition of a global discriminator for structure  
- Self-attention within U-Net bottleneck  
- Perceptual VGG-based similarity loss  
- Sobel edge alignment loss  
- Spectral normalization in discriminators  

## Project Structure
```
advanced-pix2pix/
├── models/
│   ├── generator.py
│   ├── attention.py
│   ├── discriminator_patch.py
│   ├── discriminator_global.py
├── losses/
│   ├── gan_loss.py
│   ├── perceptual_loss.py
│   └── edge_loss.py
├── dataset.py
├── train.py
├── eval_fid_lpips.py
├── utils.py
└── README.md
```

## Installation
```
pip install -r requirements.txt
```

**Dependencies:**
-- torch  
-- torchvision  
-- pillow  
-- tqdm  
-- numpy  
-- pytorch-fid  
-- lpips  

## Dataset
**Download Maps dataset:**
```
wget http://efrosgans.eecs.berkeley.edu/pix2pix/datasets/maps.tar.gz
tar -xvf maps.tar.gz
```

**Expected structure:**
```
maps/
 ├── train/
 └── val/
```

## Training
```
python train.py
```

**Default behavior:**
- trains for 100 epochs  
- saves sample generations every 10 epochs in `outputs/`  
- checkpoints stored automatically  

## Evaluation
```
python eval_fid_lpips.py
```

**Results Obtained:**
- FID ≈ 120  
- LPIPS ≈ 0.30  

## Metric Progression
- Epoch 1 → FID ~340 / LPIPS ~0.65  
- Epoch 20 → FID ~190 / LPIPS ~0.36  
- Epoch 50 → FID ~140 / LPIPS ~0.30  
- Epoch 100 → FID ~90–120 / LPIPS ~0.25  

## Observed Improvements
- better structure in generated satellite maps  
- sharper road/building edges  
- reduced blur  
- improved texture realism  

## Ablation Suggestions
Disable per experiment:  
- attention  
- global discriminator  
- perceptual loss  
- edge loss  

**Expected outcome:**
- higher FID  
- higher LPIPS  
- degraded visual quality  

## Applications
- satellite and aerial imagery translation  
- medical modality mapping (CT → MRI)  
- architectural renderings  
- segmentation → photo synthesis  
- sketch → photo generation  



## Citation

Based on: Isola et al., "Image-to-Image Translation with Conditional Adversarial Networks", 2017.
Extended via multi-scale discrimination, attention, perceptual and edge losses.
