# Brain Tumor Classification with Swin Transformer V2 and SimMIM

This repository contains code for training a Swin Transformer V2 model using SimMIM (Simple Masked Image Modeling) pre-training on an unlabelled brain tumor dataset, followed by fine-tuning for brain tumor classification.

## Overview

The training process consists of two main steps:
1. **Self-supervised pre-training** using SimMIM, which masks portions of the input images and trains the model to reconstruct them
2. **Supervised fine-tuning** for brain tumor classification using the pre-trained model

## Dataset Structure

The brain tumor dataset is organized as follows:
```
brain-tumor-dataset/
├── Training/
│   ├── glioma/
│   ├── meningioma/
│   ├── notumor/
│   └── pituitary/
└── Testing/
    ├── glioma/
    ├── meningioma/
    ├── notumor/
    └── pituitary/
```

## Files Description

- `brain_tumor_dataset.py`: Custom dataset loaders for SimMIM pre-training and classification
- `swinv2_base_simmim_pt_config.py`: Configuration for SimMIM pre-training
- `swinv2_base_simmim_ft_config.py`: Configuration for fine-tuning
- `simmim_pretrain.py`: Main script for SimMIM pre-training
- `simmim_finetune.py`: Main script for fine-tuning the model for classification

## Requirements

- PyTorch >= 1.7.0
- torchvision
- timm
- numpy
- Pillow

## Pre-training with SimMIM

To pre-train the Swin Transformer V2 model using SimMIM:

```bash
python simmim_pretrain.py --batch-size 32 --epochs 100 --output output/simmim_pretrain --tag swinv2_base_pt
```

This will train the model to reconstruct masked portions of the brain tumor images, learning useful representations without requiring labels.

### Apple Silicon GPU Support

If you're using a Mac with Apple Silicon (M1/M2/M3), you can utilize the Metal Performance Shaders (MPS) backend for accelerated training:

```bash
python simmim_pretrain.py --batch-size 32 --epochs 100 --device mps
```

For CPU-only training:

```bash
python simmim_pretrain.py --batch-size 16 --epochs 100 --device cpu
```

## Fine-tuning for Classification

After pre-training, fine-tune the model for brain tumor classification:

```bash
python simmim_finetune.py --batch-size 32 --epochs 50 --pretrained output/simmim_pretrain/swinv2_base_pt/ckpt_epoch_99.pth --output output/simmim_finetune --tag swinv2_base_ft
```

With Apple Silicon:

```bash
python simmim_finetune.py --batch-size 32 --epochs 50 --pretrained output/simmim_pretrain/swinv2_base_pt/ckpt_epoch_99.pth --device mps
```

Replace the path to the pre-trained checkpoint with your actual checkpoint path.

## Evaluation

To evaluate a trained model:

```bash
python simmim_finetune.py --eval --pretrained output/simmim_finetune/swinv2_base_ft/ckpt_epoch_49.pth
```

## Training Parameters

You can adjust various training parameters in the configuration files:

- Model architecture: Modify `swinv2_base_simmim_pt_config.py` and `swinv2_base_simmim_ft_config.py`
- Learning rate, batch size, epochs: Pass as command-line arguments or modify the config files
- Data augmentation: Adjust in the config files

## Distributed Training

For multi-GPU training, use the PyTorch distributed launch utility:

```bash
python -m torch.distributed.launch --nproc_per_node=N simmim_pretrain.py
```

Replace `N` with the number of GPUs you want to use.

## References

- [Swin Transformer V2](https://github.com/microsoft/Swin-Transformer)
- [SimMIM: A Simple Framework for Masked Image Modeling](https://arxiv.org/abs/2111.09886)
