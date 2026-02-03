"""
Inference script for SPI (Single-Pixel Imaging) with fixed patterns

Evaluates reconstruction quality using Hadamard or Random binary patterns.

Usage:
    python inference.py --checkpoint checkpoints/best_model.pth --data_dir cyto128
"""

import os
import argparse
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from tqdm import tqdm

from model import SPIModel, create_spi_model
from losses import SSIM, compute_psnr
from utils import CytoImageDataset


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SPI inference with fixed patterns')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='cyto128',
                        help='Path to dataset directory')
    parser.add_argument('--split', type=str, default='test',
                        help='Dataset split to use (train/valid/test)')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size (H=W)')
    
    # Model
    parser.add_argument('--n_measurements', type=int, default=1024,
                        help='Number of measurements')
    parser.add_argument('--noise_std', type=float, default=0.05,
                        help='Noise standard deviation')
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base features for U-Net')
    parser.add_argument('--pattern_type', type=str, default='hadamard',
                        choices=['hadamard', 'random'],
                        help='Pattern type')
    
    # Checkpoint
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='results',
                        help='Directory to save results')
    parser.add_argument('--num_samples', type=int, default=10,
                        help='Number of samples to visualize')
    parser.add_argument('--save_all', action='store_true',
                        help='Save all reconstructed images')
    
    # Inference options
    parser.add_argument('--add_noise', action='store_true',
                        help='Add noise during inference')
    
    return parser.parse_args()


class SPIInference:
    """
    SPI model inference class.
    """
    def __init__(self, model, device='cuda'):
        self.model = model
        self.device = device
        self.model.eval()
    
    @torch.no_grad()
    def reconstruct(self, x, add_noise=False):
        """
        Reconstruct image from measurements.
        
        Args:
            x: Input image (B, 1, H, W)
            add_noise: Whether to add noise
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        x = x.to(self.device)
        
        # Get measurements
        measurements = self.model.encode(x, add_noise=add_noise)
        
        # Reconstruct
        reconstructed = self.model.decode(measurements)
        
        return reconstructed
    
    @torch.no_grad()
    def reconstruct_from_measurements(self, measurements):
        """
        Reconstruct from raw measurements.
        
        Args:
            measurements: Measurements tensor (B, m)
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        measurements = measurements.to(self.device)
        return self.model.decode(measurements)
    
    def get_pattern_statistics(self):
        """Get statistics about the fixed patterns."""
        patterns = self.model.get_patterns().cpu().numpy()
        
        stats = {
            'shape': patterns.shape,
            'min': patterns.min(),
            'max': patterns.max(),
            'plus_one_ratio': (patterns == 1).mean(),
            'minus_one_ratio': (patterns == -1).mean(),
            'pattern_type': self.model.pattern_type
        }
        return stats


def evaluate_reconstruction(original, reconstructed, ssim_fn):
    """
    Compute reconstruction metrics.
    
    Returns:
        dict with PSNR and SSIM values
    """
    psnr = compute_psnr(reconstructed, original)
    ssim = ssim_fn(reconstructed, original).item()
    
    return {'psnr': psnr, 'ssim': ssim}


def visualize_results(images, reconstructed, save_path, n_samples=5):
    """
    Visualize original and reconstructed images.
    
    Args:
        images: Original images (B, 1, H, W)
        reconstructed: Reconstructed images (B, 1, H, W)
        save_path: Path to save figure
        n_samples: Number of samples to show
    """
    n_samples = min(n_samples, images.shape[0])
    
    fig, axes = plt.subplots(2, n_samples, figsize=(3*n_samples, 6))
    
    for i in range(n_samples):
        # Original
        axes[0, i].imshow(images[i, 0].cpu().numpy(), cmap='gray')
        axes[0, i].set_title('Original')
        axes[0, i].axis('off')
        
        # Reconstructed
        axes[1, i].imshow(reconstructed[i, 0].cpu().numpy(), cmap='gray')
        psnr = compute_psnr(reconstructed[i:i+1], images[i:i+1])
        axes[1, i].set_title(f'Recon (PSNR: {psnr:.2f})')
        axes[1, i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved visualization to {save_path}")


def visualize_patterns(patterns, save_path, n_patterns=16):
    """
    Visualize measurement patterns.
    
    Args:
        patterns: Pattern matrix (m, n)
        save_path: Path to save figure
        n_patterns: Number of patterns to visualize
    """
    n_patterns = min(n_patterns, patterns.shape[0])
    n_cols = 4
    n_rows = (n_patterns + n_cols - 1) // n_cols
    
    img_size = int(np.sqrt(patterns.shape[1]))
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
    axes = axes.flatten()
    
    for i in range(n_patterns):
        pattern = patterns[i].reshape(img_size, img_size)
        if isinstance(pattern, torch.Tensor):
            pattern = pattern.cpu().numpy()
        
        axes[i].imshow(pattern, cmap='gray', vmin=-1, vmax=1)
        axes[i].set_title(f'Pattern {i+1}')
        axes[i].axis('off')
    
    # Hide unused axes
    for i in range(n_patterns, len(axes)):
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"Saved pattern visualization to {save_path}")


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint and get config
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        saved_args = checkpoint.get('args', {})
        
        # Use saved args if available
        pattern_type = saved_args.get('pattern_type', args.pattern_type)
        n_measurements = saved_args.get('n_measurements', args.n_measurements)
        base_features = saved_args.get('base_features', args.base_features)
        noise_std = saved_args.get('noise_std', args.noise_std)
    else:
        pattern_type = args.pattern_type
        n_measurements = args.n_measurements
        base_features = args.base_features
        noise_std = args.noise_std
    
    # Create model
    model = create_spi_model(
        img_size=args.img_size,
        n_measurements=n_measurements,
        noise_std=noise_std,
        base_features=base_features,
        pattern_type=pattern_type
    ).to(device)
    
    # Load weights (best_model.pth only contains generator weights)
    if os.path.exists(args.checkpoint):
        state_dict = checkpoint['generator_state_dict']
        # Check if it's generator-only or full model checkpoint
        first_key = list(state_dict.keys())[0]
        if first_key.startswith('generator.') or first_key.startswith('encoder.'):
            # Full model checkpoint
            model.load_state_dict(state_dict)
        else:
            # Generator-only checkpoint (from best_model.pth)
            model.generator.load_state_dict(state_dict)
        print(f"Loaded checkpoint from {args.checkpoint}")
        print(f"  Epoch: {checkpoint.get('epoch', 'unknown')}")
        print(f"  Best PSNR: {checkpoint.get('best_psnr', 'unknown')}")
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")
    
    # Create inference object
    inference = SPIInference(model, device)
    
    # Print pattern statistics
    stats = inference.get_pattern_statistics()
    print(f"\nPattern Statistics:")
    print(f"  Type: {stats['pattern_type']}")
    print(f"  Shape: {stats['shape']}")
    print(f"  +1 ratio: {stats['plus_one_ratio']:.4f}")
    print(f"  -1 ratio: {stats['minus_one_ratio']:.4f}")
    
    # Visualize patterns
    patterns = model.get_patterns()
    visualize_patterns(patterns, os.path.join(args.output_dir, 'patterns.png'))
    
    # Load dataset
    data_path = os.path.join(args.data_dir, args.split)
    dataset = CytoImageDataset(data_path, img_size=args.img_size)
    dataloader = torch.utils.data.DataLoader(
        dataset, batch_size=32, shuffle=False, num_workers=4
    )
    
    print(f"\nEvaluating on {len(dataset)} images from {args.split} split...")
    
    # Evaluate
    ssim_fn = SSIM().to(device)
    all_psnr = []
    all_ssim = []
    
    all_images = []
    all_recon = []
    
    for images in tqdm(dataloader, desc='Evaluating'):
        images = images.to(device)
        
        # Reconstruct
        reconstructed = inference.reconstruct(images, add_noise=args.add_noise)
        
        # Compute metrics
        for i in range(images.shape[0]):
            metrics = evaluate_reconstruction(
                images[i:i+1], reconstructed[i:i+1], ssim_fn
            )
            all_psnr.append(metrics['psnr'])
            all_ssim.append(metrics['ssim'])
        
        # Store for visualization
        if len(all_images) * 32 < args.num_samples:
            all_images.append(images.cpu())
            all_recon.append(reconstructed.cpu())
    
    # Print results
    print(f"\nResults:")
    print(f"  PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB")
    print(f"  SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}")
    
    # Visualize
    if all_images:
        images_cat = torch.cat(all_images, dim=0)[:args.num_samples]
        recon_cat = torch.cat(all_recon, dim=0)[:args.num_samples]
        visualize_results(
            images_cat, recon_cat,
            os.path.join(args.output_dir, 'reconstruction_results.png'),
            n_samples=min(args.num_samples, 10)
        )
    
    # Save all reconstructions if requested
    if args.save_all:
        print("\nSaving all reconstructions...")
        recon_dir = os.path.join(args.output_dir, 'reconstructions')
        os.makedirs(recon_dir, exist_ok=True)
        
        idx = 0
        for images in tqdm(dataloader, desc='Saving'):
            images = images.to(device)
            reconstructed = inference.reconstruct(images, add_noise=args.add_noise)
            
            for i in range(images.shape[0]):
                # Save original
                orig_img = (images[i, 0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(orig_img).save(
                    os.path.join(recon_dir, f'{idx:04d}_original.png')
                )
                
                # Save reconstruction
                recon_img = (reconstructed[i, 0].cpu().numpy() * 255).astype(np.uint8)
                Image.fromarray(recon_img).save(
                    os.path.join(recon_dir, f'{idx:04d}_reconstructed.png')
                )
                
                idx += 1
        
        print(f"Saved {idx} image pairs to {recon_dir}")
    
    # Save metrics
    metrics_path = os.path.join(args.output_dir, 'metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write(f"Pattern Type: {pattern_type}\n")
        f.write(f"Measurements: {n_measurements}\n")
        f.write(f"Noise Std: {noise_std}\n")
        f.write(f"Add Noise: {args.add_noise}\n")
        f.write(f"\nResults on {args.split} set:\n")
        f.write(f"PSNR: {np.mean(all_psnr):.2f} ± {np.std(all_psnr):.2f} dB\n")
        f.write(f"SSIM: {np.mean(all_ssim):.4f} ± {np.std(all_ssim):.4f}\n")
    print(f"Saved metrics to {metrics_path}")
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
