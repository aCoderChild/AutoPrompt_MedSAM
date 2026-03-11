# MedSAM: Foundation Model for Medical Image Segmentation

This folder contains inference scripts for MedSAM, the foundation model that this project is built upon. MedSAM can be used zero-shot or fine-tuned for specific segmentation tasks.

## Overview

MedSAM is a foundation model trained on large-scale medical imaging data. It can generalize to new anatomical structures and modalities with minimal or no task-specific training. This makes it an excellent baseline for medical image segmentation.

## Prerequisites

Install required dependencies:

```bash
pip install torch torchvision
pip install segment-anything
# For MedSAM specifically
git clone https://github.com/bowang-lab/MedSAM.git
cd MedSAM
pip install -e .
```

## Architecture Features

- **Vision Transformer Backbone**: ViT-H/L/B encoder
- **Prompt-based Architecture**: Supports various prompt types (boxes, points, text)
- **Zero-shot Capability**: Works on unseen anatomies without fine-tuning
- **Flexible Prompts**: Bounding boxes, points, or masks as input
- **Foundation Model**: Pre-trained on diverse medical imaging datasets

## Model Variants

- **ViT-H**: Largest model, best accuracy but slower
- **ViT-L**: Medium-sized, good balance
- **ViT-B**: Smallest, fastest inference

## Inference

### Zero-shot (No Fine-tuning)

For zero-shot segmentation using bounding box prompts:

```bash
python infer_medsam_2D.py \
    -checkpoint /path/to/medsam_vit_b.pt \
    -data_root /path/to/input \
    -pred_save_dir /path/to/output \
    --prompt_type box
```

### 2D Images

```bash
python infer_medsam_2D.py \
    -checkpoint /path/to/medsam_best.pt \
    -data_root /path/to/input \
    -pred_save_dir /path/to/output \
    --save_overlay \
    -png_save_dir /path/to/overlay
```

### 3D Images (Stack-wise)

```bash
python infer_medsam_3D.py \
    -checkpoint /path/to/medsam_best.pt \
    -data_root /path/to/input \
    -pred_save_dir /path/to/output \
    --save_overlay \
    -png_save_dir /path/to/overlay
```

## Fine-tuning on Custom Data

To fine-tune MedSAM on your specific medical imaging task:

```bash
python finetune_medsam.py \
    -i /path/to/training/data \
    -o /path/to/output \
    -pretrain /path/to/medsam_vit_b.pt \
    --max_epochs 100 \
    -batch_size 4
```

## Data Format

Expects input data in `npz` format from MedSAM's preprocessing pipeline.

## Pre-trained Models

Download pre-trained MedSAM checkpoints:

- [MedSAM ViT-B](https://drive.google.com/drive/folders/1xYUgdjIsmBkobiBKXNb1uyqN-kGHW2p_?usp=sharing)

## References

- Ma, J., Wang, B., Li, X., et al. (2024). "Segment Anything in Medical Images"
- [MedSAM GitHub](https://github.com/bowang-lab/MedSAM)
- [SAM Paper](https://arxiv.org/abs/2304.02643)

## Key Advantages

- **Zero-shot Capability**: Works without retraining on new tasks
- **Generalization**: Strong performance on unseen anatomies
- **Flexibility**: Can be fine-tuned for better performance
- **Robustness**: Works across multiple modalities (CT, MRI, X-ray, etc.)
- **Strong Baseline**: Excellent reference point for medical segmentation
