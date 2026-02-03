# Data-driven SPI: Reconstruction and Deployment

Single-pixel imaging (SPI) reconstruction with deep learning and model pruning for efficient deployment.

## Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/Xia-Research-Lab/Data-driven-SPI.git
cd Data-driven-SPI

# Create conda environment (recommended)
conda create -n spi python=3.11
conda activate spi

# Install dependencies
pip install -r requirements.txt
```

### Basic Usage

```bash
# 1. Train the model from scratch
python train.py --epochs 100 --batch_size 16 --learning_rate 1e-3

# 2. Prune and compress the trained model (50% sparsity, 3.29× compression)
python deploy_pruned.py --checkpoint checkpoints/best_model.pth \
                         --prune_ratio 0.5 \
                         --prune_layers all \
                         --finetune_epochs 5

# 3. Run inference with the compressed model
python inference_for_deploy.py --model_path pruned_output/generator_pruned_compressed.pth.gz \
                               --benchmark \
                               --num_benchmark_runs 100
```

**Expected Results**:
- Training PSNR: ~22.39 dB
- After 50% pruning + fine-tuning: ~22.84 dB
- Inference speed: 324.7 FPS (GPU)
- Model size: 194 MB → 59 MB (3.29× compression)

## Overview

- **Input**: 2048 fixed binary measurements (12.5% sampling rate)
- **Output**: 128×128 grayscale images
- **Architecture**: U-Net based generator from measurements
- **Optimization**: 50% pruning + 3.29× compression for deployment
- **PTQ**: INT4/INT8 PTQ for real world deployment

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
├── model.py                      # Generator & Discriminator architecture
├── losses.py                     # Training losses (binary regularization, adversarial)
├── train.py                      # Training script
├── deploy_pruned.py              # Pruning & compression pipeline
├── inference_for_deploy.py       # Optimized inference for compressed models
├── ptq.py                        # Post-Training Quantization (INT8/INT4)
├── checkpoints/                  # Trained models (194 MB)
├── pruned_output/                # Pruned & compressed models (59 MB)
├── ptq_output/                   # Quantized models
├── cyto128/                      # Training dataset
├── assets/                       # Visualizations & results
├── requirements.txt              # Python dependencies
└── README.md                     # This file
```

## Requirements

```txt
torch==2.9.1
torchvision==0.24.1
pytorch-lightning==2.6.0
pytorch_ssim==0.1
numpy==2.4.1
Pillow>=10.0.0
scikit-image==0.25.2
scikit-learn==1.8.0
torchmetrics==1.8.2
torchsummary==1.5.1
```

### Installation

```bash
pip install -r requirements.txt
```

Or with conda:

```bash
conda create -n spi python=3.11 pytorch::pytorch pytorch::pytorch-cuda=12.1 pytorch::torchvision -c pytorch -c nvidia
pip install pytorch-lightning pytorch_ssim scikit-image scikit-learn torchmetrics torchsummary
```