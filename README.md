# Data-driven SPI: Reconstruction and Deployment

Single-pixel imaging (SPI) reconstruction with deep learning and model pruning for efficient deployment.

## Overview

- **Input**: 2048 fixed binary measurements (12.5% sampling rate)
- **Output**: 128×128 grayscale images
- **Architecture**: U-Net based generator from measurements
- **Optimization**: 50% pruning + 3.29× compression for deployment

## Fixed Measurement Patterns

![Patterns](./assets/patterns_visualization.png)

Pre-generated Hadamard/Random binary patterns for consistent measurements.

## Reconstruction Examples

| Original | Measurements | Reconstructed |
|:---:|:---:|:---:|
| ![Original](./assets/original_image.png) | ![Reconstruction](./assets/reconstruction_result.png) | ![Sampled](./assets/sampled_measurements_0000_reconstructed.png) |

**Performance**: PSNR ~22-27 dB | SSIM ~0.65-0.85

## Training

```bash
python train.py --epochs 100 --batch_size 16 --learning_rate 1e-3
```

## Model Inference

### Original Model (FP32)
```bash
python inference.py --checkpoint checkpoints/best_model.pth --image_path test.png
```

### Pruned Model (Compressed)
```bash
python inference_after_prune.py --model_path pruned_output/generator_pruned_compressed.pth.gz \
                                 --measurement_file measurements.npy
```

## Model Compression

Pruning reduces model size while maintaining performance through fine-tuning.

![Pruning](./assets/pruning_visualization.png)

| Metric | Value |
|--------|-------|
| Original Size | 194 MB |
| Compressed Size | 59 MB |
| Compression Ratio | 3.29× |
| Sparsity | 49.98% |
| PSNR (original→pruned→finetuned) | 22.39 → 21.44 → 22.84 dB |
| Inference Speed | 324.7 FPS |

## Pruning & Fine-tuning

```bash
python prune.py --checkpoint checkpoints/best_model.pth \
                 --prune_ratio 0.5 \
                 --finetune_epochs 5 \
                 --save_sparse
```

## Project Structure

```
.
├── model.py                    # Generator architecture
├── losses.py                   # Training losses
├── train.py                    # Training script
├── prune.py                    # Pruning & compression
├── inference.py                # FP32 inference
├── inference_after_prune.py    # Compressed model inference
├── checkpoints/                # Trained models (194 MB)
├── pruned_output/              # Compressed models (59 MB)
├── cyto128/                    # Dataset
└── assets/                     # Visualizations
```

## Requirements

```
torch>=1.9.0
torchvision
numpy
Pillow
```