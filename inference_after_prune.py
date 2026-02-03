"""
Inference script for Pruned SPI Generator

This script loads a pruned and exported generator model for inference.
Designed for deployment scenarios where:
1. Patterns are pre-loaded on DMD
2. Measurements come from hardware or simulation
3. Fast reconstruction is needed

Usage:
    # From measurement file
    python inference_for_deploy.py --model_path deploy_pruned_output/generator_pruned_full.pth \
                                   --measurement_file measurements.npy \
                                   --output_dir inference_output

    # From image (simulated measurement)
    python inference_for_deploy.py --model_path deploy_pruned_output/generator_pruned_full.pth \
                                   --image_path test_image.png \
                                   --output_dir inference_output

    # Batch inference from directory
    python inference_for_deploy.py --model_path deploy_pruned_output/generator_pruned_full.pth \
                                   --image_dir test_images/ \
                                   --output_dir inference_output
"""

import os
import argparse
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from pathlib import Path

from model import Generator
from utils import Mish


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Inference with Pruned SPI Generator')
    
    # Model
    parser.add_argument('--model_path', default='assets/spi_pruned.pth.gz',type=str,
                        help='Path to pruned model checkpoint')
    parser.add_argument('--patterns_path', type=str, default=None,
                        help='Path to patterns file (optional, will look in same dir as model)')
    
    # Input options (mutually exclusive)
    #input_group = parser.add_mutually_exclusive_group(required=True)
    parser.add_argument('--measurement_file',default='assets/measurements_2.npy', type=str,
                            help='Path to 1D measurement data (.npy)')
    parser.add_argument('--image_path', type=str,
                            help='Path to input image (will simulate measurement)')
    parser.add_argument('--image_dir', type=str,
                            help='Directory of images for batch processing')
    
    # Output
    parser.add_argument('--output_dir', type=str, default='inference_output',
                        help='Directory to save outputs')
    parser.add_argument('--save_numpy', action='store_true',
                        help='Also save reconstruction as numpy array')
    
    # Inference options
    parser.add_argument('--noise_std', type=float, default=0.05,
                        help='Add noise to simulated measurements')
    parser.add_argument('--batch_size', type=int, default=1,
                        help='Batch size for batch inference')
    parser.add_argument('--benchmark', action='store_true',
                        help='Run inference speed benchmark')
    parser.add_argument('--num_benchmark_runs', type=int, default=100,
                        help='Number of runs for benchmark')
    
    # Device
    parser.add_argument('--device', type=str, default='auto',
                        choices=['auto', 'cuda', 'cpu'],
                        help='Device to use')
    
    return parser.parse_args()


class PrunedGeneratorInference:
    """
    Inference class for pruned SPI Generator.
    
    Optimized for deployment with minimal dependencies.
    """
    
    def __init__(self, model_path, patterns_path=None, device='auto'):
        """
        Initialize inference engine.
        
        Args:
            model_path: Path to pruned model checkpoint
            patterns_path: Path to patterns file (optional)
            device: 'auto', 'cuda', or 'cpu'
        """
        # Set device
        if device == 'auto':
            self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)
        
        print(f"Inference device: {self.device}")
        
        # Load model
        self._load_model(model_path)
        
        # Load patterns
        self._load_patterns(model_path, patterns_path)
        
        print(f"PrunedGeneratorInference initialized:")
        print(f"  Image size: {self.img_size}x{self.img_size}")
        print(f"  Measurements: {self.n_measurements}")
    
    def _load_model(self, model_path):
        """Load the pruned generator model (supports .pth, .pth.gz formats)."""
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        # Handle compressed format
        if model_path.endswith('.pth.gz'):
            import gzip
            import pickle
            print(f"Loading compressed model from: {model_path}")
            with gzip.open(model_path, 'rb') as f:
                checkpoint = pickle.load(f)
            # Compressed format is just state_dict in FP16
            state_dict = {k: v.float() for k, v in checkpoint.items()}
            # Use default config for compressed models
            self.n_measurements = 2048
            self.img_size = 128
            self.n_pixels = self.img_size * self.img_size
            self.base_features = 64
        else:
            checkpoint = torch.load(model_path, map_location=self.device)
            
            # Get model config
            if 'model_config' in checkpoint:
                config = checkpoint['model_config']
                self.n_measurements = config['n_measurements']
                self.n_pixels = config['n_pixels']
                self.img_size = config['img_size']
                self.base_features = config['base_features']
            else:
                # Default config
                self.n_measurements = 1024
                self.img_size = 128
                self.n_pixels = self.img_size * self.img_size
                self.base_features = 64
            
            # Load weights
            if 'generator_state_dict' in checkpoint:
                state_dict = checkpoint['generator_state_dict']
            else:
                state_dict = checkpoint
        
        # Create generator
        self.generator = Generator(
            n_measurements=self.n_measurements,
            n_pixels=self.n_pixels,
            img_size=self.img_size,
            base_features=self.base_features
        ).to(self.device)
        
        self.generator.load_state_dict(state_dict, strict=False)
        self.generator.eval()
        
        # Count parameters
        total_params = sum(p.numel() for p in self.generator.parameters())
        nonzero_params = sum((p != 0).sum().item() for p in self.generator.parameters())
        sparsity = 1 - nonzero_params / total_params
        
        print(f"Loaded model from: {model_path}")
        print(f"  Total parameters: {total_params:,}")
        print(f"  Non-zero parameters: {nonzero_params:,}")
        print(f"  Sparsity: {sparsity:.2%}")
        
        # Store prune info if available
        self.prune_info = checkpoint.get('prune_info', {}) if isinstance(checkpoint, dict) else {}
    
    def _load_patterns(self, model_path, patterns_path):
        """Load measurement patterns."""
        # Try to find patterns
        if patterns_path is None:
            # Look in same directory as model
            model_dir = os.path.dirname(model_path)
            patterns_path = os.path.join(model_dir, 'patterns.npy')
        
        if os.path.exists(patterns_path):
            patterns = np.load(patterns_path)
            self.patterns = torch.from_numpy(patterns).float().to(self.device)
            print(f"Loaded patterns from: {patterns_path}")
            print(f"  Shape: {self.patterns.shape}")
        else:
            # Generate default Hadamard patterns
            from model import generate_hadamard_zigzag_patterns
            patterns = generate_hadamard_zigzag_patterns(self.img_size, self.n_measurements)
            self.patterns = torch.from_numpy(patterns).float().to(self.device)
            print(f"Generated default Hadamard patterns")
            print(f"  Shape: {self.patterns.shape}")
    
    def simulate_measurement(self, image, noise_std=0.0):
        """
        Simulate measurement from image.
        
        Args:
            image: Tensor of shape (B, 1, H, W) or (1, H, W) or (H, W)
            noise_std: Standard deviation of measurement noise
        
        Returns:
            Measurements tensor of shape (B, m)
        """
        # Handle different input shapes
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]
        image = image.to(self.device)
        
        # Flatten and compute measurements
        x_flat = image.view(batch_size, -1)
        measurements = F.linear(x_flat, self.patterns)
        
        # Add noise
        if noise_std > 0:
            noise = torch.randn_like(measurements) * noise_std
            measurements = measurements + noise
        
        return measurements
    
    @torch.no_grad()
    def reconstruct(self, measurements):
        """
        Reconstruct image from measurements.
        
        Args:
            measurements: Tensor of shape (B, m) or (m,)
        
        Returns:
            Reconstructed image tensor of shape (B, 1, H, W)
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
        
        measurements = measurements.to(self.device)
        reconstructed = self.generator(measurements)
        
        return reconstructed
    
    @torch.no_grad()
    def reconstruct_from_numpy(self, measurements_np):
        """
        Reconstruct from numpy measurements.
        
        Args:
            measurements_np: Numpy array of shape (m,) or (B, m)
        
        Returns:
            Reconstructed image as numpy array (H, W) or (B, H, W)
        """
        measurements = torch.from_numpy(measurements_np).float()
        reconstructed = self.reconstruct(measurements)
        
        result = reconstructed.cpu().numpy()
        if result.shape[0] == 1:
            return result[0, 0]  # (H, W)
        else:
            return result[:, 0]  # (B, H, W)
    
    @torch.no_grad()
    def reconstruct_from_image(self, image, noise_std=0.0):
        """
        Full pipeline: image -> measurement -> reconstruction.
        
        Args:
            image: Tensor or numpy array (H, W) or (B, 1, H, W)
            noise_std: Measurement noise
        
        Returns:
            Reconstructed image tensor
        """
        if isinstance(image, np.ndarray):
            image = torch.from_numpy(image).float()
        
        measurements = self.simulate_measurement(image, noise_std)
        reconstructed = self.reconstruct(measurements)
        
        return reconstructed, measurements
    
    def benchmark(self, batch_size=1, num_runs=100, warmup=10):
        """
        Benchmark inference speed.
        
        Args:
            batch_size: Batch size for benchmark
            num_runs: Number of inference runs
            warmup: Number of warmup runs
        
        Returns:
            Dictionary with timing statistics
        """
        print(f"\nBenchmarking inference speed...")
        print(f"  Batch size: {batch_size}")
        print(f"  Num runs: {num_runs}")
        
        # Create dummy input
        dummy_measurements = torch.randn(batch_size, self.n_measurements).to(self.device)
        
        # Warmup
        for _ in range(warmup):
            _ = self.reconstruct(dummy_measurements)
        
        # Synchronize if using CUDA
        if self.device.type == 'cuda':
            torch.cuda.synchronize()
        
        # Benchmark
        times = []
        for _ in range(num_runs):
            start = time.perf_counter()
            _ = self.reconstruct(dummy_measurements)
            if self.device.type == 'cuda':
                torch.cuda.synchronize()
            end = time.perf_counter()
            times.append(end - start)
        
        times = np.array(times) * 1000  # Convert to ms
        
        results = {
            'batch_size': batch_size,
            'mean_ms': np.mean(times),
            'std_ms': np.std(times),
            'min_ms': np.min(times),
            'max_ms': np.max(times),
            'median_ms': np.median(times),
            'throughput_fps': batch_size * 1000 / np.mean(times),
        }
        
        print(f"\nBenchmark Results:")
        print(f"  Mean: {results['mean_ms']:.2f} ms")
        print(f"  Std: {results['std_ms']:.2f} ms")
        print(f"  Min: {results['min_ms']:.2f} ms")
        print(f"  Max: {results['max_ms']:.2f} ms")
        print(f"  Throughput: {results['throughput_fps']:.1f} FPS")
        
        return results


def load_image(image_path, img_size):
    """Load and preprocess an image."""
    from torchvision import transforms
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img)  # (1, H, W)
    
    return img_tensor


def process_single_image(inference_engine, image_path, output_dir, noise_std=0.0, save_numpy=False):
    """Process a single image."""
    img_size = inference_engine.img_size
    
    # Load image
    image = load_image(image_path, img_size)
    image = image.unsqueeze(0)  # (1, 1, H, W)
    
    # Reconstruct
    reconstructed, measurements = inference_engine.reconstruct_from_image(image, noise_std)
    
    # Convert to numpy
    original_np = image.squeeze().numpy()
    reconstructed_np = reconstructed.squeeze().cpu().numpy()
    measurements_np = measurements.squeeze().cpu().numpy()
    
    # Compute metrics
    from losses import compute_psnr, SSIM
    psnr = compute_psnr(reconstructed, image.to(inference_engine.device))
    ssim_fn = SSIM().to(inference_engine.device)
    ssim = ssim_fn(reconstructed, image.to(inference_engine.device)).item()
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(image_path).stem
    
    # Save reconstructed image
    recon_path = os.path.join(output_dir, f'{base_name}_reconstructed.png')
    recon_uint8 = (np.clip(reconstructed_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(recon_uint8).save(recon_path)
    
    # Save numpy if requested
    if save_numpy:
        np.save(os.path.join(output_dir, f'{base_name}_reconstructed.npy'), reconstructed_np)
        np.save(os.path.join(output_dir, f'{base_name}_measurements.npy'), measurements_np)
    
    # Create visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    axes[0].imshow(original_np, cmap='gray', vmin=0, vmax=1)
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].plot(measurements_np)
    axes[1].set_title(f'Measurements (m={len(measurements_np)})')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    axes[1].grid(True, alpha=0.3)
    
    axes[2].imshow(reconstructed_np, cmap='gray', vmin=0, vmax=1)
    axes[2].set_title(f'Reconstructed\nPSNR: {psnr:.2f} dB, SSIM: {ssim:.4f}')
    axes[2].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{base_name}_comparison.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Processed: {image_path}")
    print(f"  PSNR: {psnr:.2f} dB")
    print(f"  SSIM: {ssim:.4f}")
    print(f"  Saved: {recon_path}")
    
    return {'psnr': psnr, 'ssim': ssim, 'path': recon_path}


def process_measurement_file(inference_engine, measurement_file, output_dir, save_numpy=False):
    """Process measurements from file."""
    # Load measurements
    measurements_np = np.load(measurement_file)
    print(f"Loaded measurements from: {measurement_file}")
    print(f"  Shape: {measurements_np.shape}")
    
    # Handle batch or single
    if measurements_np.ndim == 1:
        measurements_np = measurements_np[np.newaxis, :]
    
    # Reconstruct
    reconstructed_np = inference_engine.reconstruct_from_numpy(measurements_np)
    if reconstructed_np.ndim == 2:
        reconstructed_np = reconstructed_np[np.newaxis, :]
    
    # Save results
    os.makedirs(output_dir, exist_ok=True)
    base_name = Path(measurement_file).stem
    
    for i in range(reconstructed_np.shape[0]):
        recon = reconstructed_np[i]
        
        # Save image
        recon_path = os.path.join(output_dir, f'{base_name}_{i:04d}_reconstructed.png')
        recon_uint8 = (np.clip(recon, 0, 1) * 255).astype(np.uint8)
        Image.fromarray(recon_uint8).save(recon_path)
        
        if save_numpy:
            np.save(os.path.join(output_dir, f'{base_name}_{i:04d}_reconstructed.npy'), recon)
        
        print(f"  Saved: {recon_path}")
    
    # Visualization for first few
    n_show = min(4, reconstructed_np.shape[0])
    fig, axes = plt.subplots(1, n_show, figsize=(4*n_show, 4))
    if n_show == 1:
        axes = [axes]
    
    for i in range(n_show):
        axes[i].imshow(reconstructed_np[i], cmap='gray', vmin=0, vmax=1)
        axes[i].set_title(f'Sample {i+1}')
        axes[i].axis('off')
    
    plt.tight_layout()
    viz_path = os.path.join(output_dir, f'{base_name}_reconstructions.png')
    plt.savefig(viz_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    print(f"Saved visualization: {viz_path}")


def process_image_directory(inference_engine, image_dir, output_dir, noise_std=0.0, 
                           batch_size=1, save_numpy=False):
    """Process all images in a directory."""
    # Get image files
    image_extensions = {'.png', '.jpg', '.jpeg', '.tif', '.tiff', '.bmp'}
    image_files = [
        f for f in os.listdir(image_dir)
        if Path(f).suffix.lower() in image_extensions
    ]
    
    if not image_files:
        print(f"No images found in {image_dir}")
        return
    
    print(f"Found {len(image_files)} images in {image_dir}")
    
    results = []
    for img_file in image_files:
        img_path = os.path.join(image_dir, img_file)
        result = process_single_image(
            inference_engine, img_path, output_dir, 
            noise_std=noise_std, save_numpy=save_numpy
        )
        results.append(result)
    
    # Summary
    avg_psnr = np.mean([r['psnr'] for r in results])
    avg_ssim = np.mean([r['ssim'] for r in results])
    
    print(f"\n{'='*60}")
    print(f"Batch Processing Complete")
    print(f"{'='*60}")
    print(f"  Images processed: {len(results)}")
    print(f"  Average PSNR: {avg_psnr:.2f} dB")
    print(f"  Average SSIM: {avg_ssim:.4f}")
    
    # Save summary
    summary_path = os.path.join(output_dir, 'summary.txt')
    with open(summary_path, 'w') as f:
        f.write(f"Batch Inference Summary\n")
        f.write(f"{'='*50}\n\n")
        f.write(f"Images processed: {len(results)}\n")
        f.write(f"Average PSNR: {avg_psnr:.2f} dB\n")
        f.write(f"Average SSIM: {avg_ssim:.4f}\n\n")
        f.write(f"Individual Results:\n")
        for r in results:
            f.write(f"  {Path(r['path']).stem}: PSNR={r['psnr']:.2f}, SSIM={r['ssim']:.4f}\n")
    
    print(f"  Summary saved: {summary_path}")


def main():
    args = parse_args()
    
    # Create inference engine
    inference = PrunedGeneratorInference(
        model_path=args.model_path,
        patterns_path=args.patterns_path,
        device=args.device
    )
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Run benchmark if requested
    if args.benchmark:
        inference.benchmark(
            batch_size=args.batch_size,
            num_runs=args.num_benchmark_runs
        )
    
    # Process based on input type
    if args.measurement_file:
        process_measurement_file(
            inference, 
            args.measurement_file, 
            args.output_dir,
            save_numpy=args.save_numpy
        )
    # elif args.image_path:
    #     process_single_image(
    #         inference,
    #         args.image_path,
    #         args.output_dir,
    #         noise_std=args.noise_std,
    #         save_numpy=args.save_numpy
    #     )
    # elif args.image_dir:
    #     process_image_directory(
    #         inference,
    #         args.image_dir,
    #         args.output_dir,
    #         noise_std=args.noise_std,
    #         batch_size=args.batch_size,
    #         save_numpy=args.save_numpy
    #     )
    
    print(f"\nInference complete! Results saved to: {args.output_dir}")


if __name__ == "__main__":
    main()
