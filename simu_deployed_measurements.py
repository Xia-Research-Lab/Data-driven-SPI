"""
Deployment script for SPI (Single-Pixel Imaging) with fixed patterns

This script loads only the Generator (Decoder) for real-world deployment.
The fixed patterns (-1/+1 Hadamard or Random) are loaded separately for DMD control.

In deployment:
1. Fixed patterns are pre-loaded and applied to DMD
2. Measurements are collected from the physical system
3. Generator reconstructs the image from measurements

Usage:
    python deploy.py --checkpoint checkpoints/best_model.pth --patterns_file patterns.npy
"""

import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

from model import Generator, generate_hadamard_zigzag_patterns, generate_random_binary_patterns
from utils import Mish


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SPI deployment with generator only')
    
    # Model config
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size (H=W)')
    parser.add_argument('--n_measurements', type=int, default=1024,
                        help='Number of measurements')
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base features for U-Net decoder')
    parser.add_argument('--pattern_type', type=str, default='hadamard',
                        choices=['hadamard', 'random'],
                        help='Pattern type')
    
    # Files
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--patterns_file', type=str, default=None,
                        help='Path to save/load patterns (.npy)')
    parser.add_argument('--output_dir', type=str, default='simulated_measurements',
                        help='Directory to save outputs')
    
    # Measurement data
    parser.add_argument('--measurement_file', type=str, default=None,
                        help='Path to 1D measurement intensity data (.npy). If not provided, will sample from valid dataset.')
    parser.add_argument('--data_dir', type=str, default='cyto128',
                        help='Path to dataset directory (used when no measurement_file is provided)')
    
    return parser.parse_args()


class SPIDeployment:
    """
    SPI model deployment class that only loads the Generator (Decoder).
    
    This is designed for real-world single-pixel imaging systems where:
    1. Fixed patterns are physically implemented on DMD
    2. Measurements come from the actual hardware
    3. Only the generator (decoder) is needed for reconstruction
    
    Args:
        n_measurements: Number of measurements (m)
        img_size: Image size (H = W)
        base_features: Base features for U-Net decoder
        device: Device to run on
    """
    def __init__(self, n_measurements, img_size=128, base_features=64, device='cuda'):
        self.n_measurements = n_measurements
        self.img_size = img_size
        self.n_pixels = img_size * img_size
        self.base_features = base_features
        self.device = device
        
        # Initialize generator only
        self.generator = Generator(
            n_measurements=n_measurements,
            n_pixels=self.n_pixels,
            img_size=img_size,
            base_features=base_features
        ).to(device)
        
        # Fixed patterns (loaded separately)
        self.patterns = None
        
        print(f"SPIDeployment initialized:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Measurements: {n_measurements}")
        print(f"  Generator parameters: {sum(p.numel() for p in self.generator.parameters()):,}")
    
    def load_generator_weights(self, checkpoint_path):
        """
        Load generator weights from a full SPI model checkpoint.
        
        Args:
            checkpoint_path: Path to the SPI model checkpoint
        """
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Check if it's a full model checkpoint or generator-only
        state_dict = checkpoint.get('generator_state_dict', checkpoint.get('model_state_dict', checkpoint))
        
        # Extract generator weights (prefix: 'generator.')
        generator_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('generator.'):
                new_key = key[len('generator.'):]
                generator_state_dict[new_key] = value
            elif not key.startswith('encoder.'):
                # Might be a generator-only checkpoint
                generator_state_dict[key] = value
        
        if generator_state_dict:
            self.generator.load_state_dict(generator_state_dict)
        else:
            # Try loading directly
            self.generator.load_state_dict(state_dict)
        
        self.generator.eval()
        
        print(f"Loaded generator weights from {checkpoint_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    def load_patterns(self, patterns_path):
        """
        Load fixed patterns from file.
        
        Args:
            patterns_path: Path to .npy file containing patterns (-1/+1)
        """
        self.patterns = np.load(patterns_path)
        self.patterns = torch.from_numpy(self.patterns).float().to(self.device)
        
        expected_shape = (self.n_measurements, self.n_pixels)
        if self.patterns.shape != expected_shape:
            raise ValueError(f"Pattern shape mismatch. Expected {expected_shape}, got {self.patterns.shape}")
        
        unique_vals = torch.unique(self.patterns).cpu().numpy()
        print(f"Loaded patterns from {patterns_path}")
        print(f"  Shape: {self.patterns.shape}")
        print(f"  Unique values: {unique_vals}")
        print(f"  +1 ratio: {(self.patterns == 1).float().mean().item():.4f}")
    
    def generate_and_save_patterns(self, output_path, pattern_type='hadamard', seed=42):
        """
        Generate fixed patterns and save them.
        
        Args:
            output_path: Path to save patterns (.npy)
            pattern_type: 'hadamard' or 'random'
            seed: Random seed for random patterns
        """
        if pattern_type == 'hadamard':
            patterns = generate_hadamard_zigzag_patterns(self.img_size, self.n_measurements)
        elif pattern_type == 'random':
            patterns = generate_random_binary_patterns(self.img_size, self.n_measurements, seed)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}")
        
        # Save
        np.save(output_path, patterns)
        
        # Also load into memory
        self.patterns = torch.from_numpy(patterns).float().to(self.device)
        
        print(f"Generated and saved {pattern_type} patterns to {output_path}")
        print(f"  Shape: {patterns.shape}")
        print(f"  +1 ratio: {(patterns == 1).mean():.4f}")
        print(f"  -1 ratio: {(patterns == -1).mean():.4f}")
        
        return patterns
    
    def extract_patterns_from_checkpoint(self, checkpoint_path, output_path):
        """
        Extract patterns from a checkpoint and save them.
        
        Args:
            checkpoint_path: Path to SPI model checkpoint
            output_path: Path to save patterns (.npy)
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        state_dict = checkpoint.get('generator_state_dict', checkpoint)
        
        # Look for pattern buffer in encoder
        patterns = None
        for key, value in state_dict.items():
            if 'encoder.patterns' in key or key == 'encoder.patterns':
                patterns = value.numpy()
                break
        
        if patterns is None:
            # Generate based on saved args
            saved_args = checkpoint.get('args', {})
            pattern_type = saved_args.get('pattern_type', 'hadamard')
            print(f"Patterns not found in checkpoint. Generating {pattern_type} patterns...")
            patterns = self.generate_and_save_patterns(output_path, pattern_type)
        else:
            np.save(output_path, patterns)
            self.patterns = torch.from_numpy(patterns).float().to(self.device)
            print(f"Extracted and saved patterns to {output_path}")
            print(f"  Shape: {patterns.shape}")
        
        return patterns
    
    def simulate_measurement(self, image):
        """
        Simulate measurement process using fixed patterns.
        This is for testing; in real deployment, measurements come from hardware.
        
        Args:
            image: Input image tensor (B, 1, H, W) or (1, H, W) or (H, W)
        
        Returns:
            Measurements tensor (B, m)
        """
        if self.patterns is None:
            raise RuntimeError("Patterns not loaded. Call load_patterns() first.")
        
        # Handle different input shapes
        if image.dim() == 2:
            image = image.unsqueeze(0).unsqueeze(0)
        elif image.dim() == 3:
            image = image.unsqueeze(0)
        
        batch_size = image.shape[0]
        image = image.to(self.device)
        
        # Flatten and compute measurements: y = E @ x
        x_flat = image.view(batch_size, -1)
        measurements = F.linear(x_flat, self.patterns)
        
        return measurements
    
    @torch.no_grad()
    def reconstruct(self, measurements):
        """
        Reconstruct image from measurements.
        
        Args:
            measurements: Measurements tensor (B, m) or (m,)
        
        Returns:
            Reconstructed image (B, 1, H, W)
        """
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
        
        measurements = measurements.to(self.device)
        
        # Ensure eval mode for inference
        self.generator.eval()
        reconstructed = self.generator(measurements)
        
        return reconstructed
    
    @torch.no_grad()
    def reconstruct_from_numpy(self, measurements_np):
        """
        Reconstruct from numpy measurements array.
        
        Args:
            measurements_np: Numpy array of shape (m,) or (B, m)
        
        Returns:
            Reconstructed image as numpy array (H, W) or (B, H, W)
        """
        measurements = torch.from_numpy(measurements_np).float()
        reconstructed = self.reconstruct(measurements)
        
        # Convert to numpy
        result = reconstructed.cpu().numpy()
        if result.shape[0] == 1:
            return result[0, 0]  # (H, W)
        else:
            return result[:, 0]  # (B, H, W)
    
    def visualize_patterns(self, save_path, n_patterns=16):
        """
        Visualize the fixed patterns.
        
        Args:
            save_path: Path to save figure
            n_patterns: Number of patterns to show
        """
        if self.patterns is None:
            raise RuntimeError("Patterns not loaded.")
        
        n_patterns = min(n_patterns, self.n_measurements)
        n_cols = 4
        n_rows = (n_patterns + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(3*n_cols, 3*n_rows))
        axes = axes.flatten()
        
        for i in range(n_patterns):
            pattern = self.patterns[i].reshape(self.img_size, self.img_size).cpu().numpy()
            axes[i].imshow(pattern, cmap='gray', vmin=-1, vmax=1)
            axes[i].set_title(f'Pattern {i+1}')
            axes[i].axis('off')
        
        for i in range(n_patterns, len(axes)):
            axes[i].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved pattern visualization to {save_path}")
    
    def export_patterns_for_dmd(self, output_dir, format='bmp'):
        """
        Export patterns as individual images for DMD loading.
        
        Args:
            output_dir: Directory to save pattern images
            format: Image format ('bmp', 'png')
        """
        if self.patterns is None:
            raise RuntimeError("Patterns not loaded.")
        
        os.makedirs(output_dir, exist_ok=True)
        
        for i in range(self.n_measurements):
            pattern = self.patterns[i].reshape(self.img_size, self.img_size).cpu().numpy()
            # Convert -1/+1 to 0/255 for DMD
            pattern_uint8 = ((pattern + 1) / 2 * 255).astype(np.uint8)
            
            img = Image.fromarray(pattern_uint8)
            img.save(os.path.join(output_dir, f'pattern_{i:04d}.{format}'))
        
        print(f"Exported {self.n_measurements} patterns to {output_dir}")
        
        # Also save pattern info
        info_path = os.path.join(output_dir, 'pattern_info.txt')
        with open(info_path, 'w') as f:
            f.write(f"Number of patterns: {self.n_measurements}\n")
            f.write(f"Pattern size: {self.img_size}x{self.img_size}\n")
            f.write(f"Values: -1 (black, 0 in image) and +1 (white, 255 in image)\n")
            f.write(f"Format: {format}\n")
        print(f"Saved pattern info to {info_path}")


def demo_reconstruction(deployment, test_image_path=None):
    """
    Demo reconstruction with a test image.
    
    Args:
        deployment: SPIDeployment instance
        test_image_path: Path to test image (optional)
    """
    if test_image_path and os.path.exists(test_image_path):
        # Load test image
        img = Image.open(test_image_path).convert('L')
        img = img.resize((deployment.img_size, deployment.img_size))
        img_np = np.array(img, dtype=np.float32) / 255.0
    else:
        # Create synthetic test image
        print("Creating synthetic test image...")
        x = np.linspace(-2, 2, deployment.img_size)
        y = np.linspace(-2, 2, deployment.img_size)
        X, Y = np.meshgrid(x, y)
        img_np = np.exp(-(X**2 + Y**2) / 2) * 0.8 + 0.1
    
    # Convert to tensor
    img_tensor = torch.from_numpy(img_np).float().unsqueeze(0).unsqueeze(0).to(deployment.device)
    
    # Simulate measurement
    measurements = deployment.simulate_measurement(img_tensor)
    print(f"Measurements shape: {measurements.shape}")
    print(f"Measurements range: [{measurements.min():.2f}, {measurements.max():.2f}]")
    
    # Reconstruct
    reconstructed = deployment.reconstruct(measurements)
    
    # Visualize
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    
    axes[0].imshow(img_np, cmap='gray')
    axes[0].set_title('Original')
    axes[0].axis('off')
    
    axes[1].plot(measurements[0].cpu().numpy()[:100])
    axes[1].set_title('Measurements (first 100)')
    axes[1].set_xlabel('Index')
    axes[1].set_ylabel('Value')
    
    axes[2].imshow(reconstructed[0, 0].cpu().numpy(), cmap='gray')
    axes[2].set_title('Reconstructed')
    axes[2].axis('off')
    
    plt.tight_layout()
    plt.savefig('deploy_output/demo_reconstruction.png', dpi=150)
    plt.close()
    print("Saved demo reconstruction to deploy_output/demo_reconstruction.png")
    
    # Compute metrics
    from losses import compute_psnr, SSIM
    psnr = compute_psnr(reconstructed, img_tensor)
    ssim_fn = SSIM().to(deployment.device)
    ssim = ssim_fn(reconstructed, img_tensor).item()
    print(f"PSNR: {psnr:.2f} dB")
    print(f"SSIM: {ssim:.4f}")


def load_random_valid_image(data_dir, img_size=128):
    """
    Load a random image from the valid dataset.
    
    Args:
        data_dir: Path to dataset directory containing 'valid' subfolder
        img_size: Target image size
    
    Returns:
        image_tensor: Tensor of shape (1, 1, H, W)
        image_path: Path to the loaded image
    """
    from torchvision import transforms
    import random
    
    valid_dir = os.path.join(data_dir, 'valid')
    if not os.path.exists(valid_dir):
        raise FileNotFoundError(f"Valid directory not found: {valid_dir}")
    
    # Get all image files
    image_files = [
        f for f in os.listdir(valid_dir)
        if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
    ]
    
    if not image_files:
        raise FileNotFoundError(f"No images found in {valid_dir}")
    
    # Random select
    selected_file = random.choice(image_files)
    image_path = os.path.join(valid_dir, selected_file)
    
    # Load and transform
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    img = Image.open(image_path).convert('RGB')
    img_tensor = transform(img).unsqueeze(0)  # (1, 1, H, W)
    
    return img_tensor, image_path


def main():
    args = parse_args()
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load checkpoint config
    if os.path.exists(args.checkpoint):
        checkpoint = torch.load(args.checkpoint, map_location=device)
        saved_args = checkpoint.get('args', {})
        
        n_measurements = saved_args.get('n_measurements', args.n_measurements)
        base_features = saved_args.get('base_features', args.base_features)
        pattern_type = saved_args.get('pattern_type', args.pattern_type)
    else:
        n_measurements = args.n_measurements
        base_features = args.base_features
        pattern_type = args.pattern_type
    
    # Create deployment object
    deployment = SPIDeployment(
        n_measurements=n_measurements,
        img_size=args.img_size,
        base_features=base_features,
        device=device
    )
    
    # Load generator weights
    if os.path.exists(args.checkpoint):
        deployment.load_generator_weights(args.checkpoint)
    else:
        print(f"Warning: Checkpoint {args.checkpoint} not found. Using random weights.")
    
    # Load or generate patterns
    if args.patterns_file and os.path.exists(args.patterns_file):
        deployment.load_patterns(args.patterns_file)
    else:
        patterns_path = args.patterns_file or os.path.join(args.output_dir, 'patterns.npy')
        deployment.generate_and_save_patterns(patterns_path, pattern_type)
    
    # Visualize patterns
    deployment.visualize_patterns(os.path.join(args.output_dir, 'patterns_visualization.png'))
    
    # Export for DMD
    export_dir = os.path.join(args.output_dir, 'dmd_patterns')
    print(f"\nExporting patterns for DMD to {export_dir}...")
    deployment.export_patterns_for_dmd(export_dir)
    
    # ================================================================
    # Reconstruction from 1D intensity measurements
    # ================================================================
    print("\n" + "="*60)
    print("Reconstruction from 1D intensity measurements")
    print("="*60)
    
    original_image = None
    original_path = None
    
    if args.measurement_file and os.path.exists(args.measurement_file):
        # Load measurements from file
        print(f"\nLoading measurements from: {args.measurement_file}")
        measurements_np = np.load(args.measurement_file)
        measurements = torch.from_numpy(measurements_np).float().to(device)
        
        if measurements.dim() == 1:
            measurements = measurements.unsqueeze(0)
        
        print(f"  Measurements shape: {measurements.shape}")
        print(f"  Measurements range: [{measurements.min():.4f}, {measurements.max():.4f}]")
        
    else:
        # Sample from valid dataset (no noise)
        print(f"\nNo measurement file provided. Sampling from valid dataset...")
        print(f"  Data directory: {args.data_dir}")
        
        # Load random image from valid set
        original_image, original_path = load_random_valid_image(args.data_dir, args.img_size)
        original_image = original_image.to(device)
        print(f"  Selected image: {original_path}")
        
        # Simulate measurement (dot product, no noise)
        measurements = deployment.simulate_measurement(original_image)
        measurements_np = measurements.cpu().numpy().squeeze()
        
        print(f"  Measurements shape: {measurements_np.shape}")
        print(f"  Measurements range: [{measurements_np.min():.4f}, {measurements_np.max():.4f}]")
        
        # Save measurements
        measurements_save_path = os.path.join(args.output_dir, 'sampled_measurements.npy')
        np.save(measurements_save_path, measurements_np)
        print(f"  Saved measurements to: {measurements_save_path}")
        
        # Save original image
        original_save_path = os.path.join(args.output_dir, 'original_image.png')
        original_np = original_image.cpu().numpy().squeeze()
        Image.fromarray((original_np * 255).astype(np.uint8)).save(original_save_path)
        print(f"  Saved original image to: {original_save_path}")
    
    # Reconstruct using decoder
    print("\nReconstructing image using decoder...")
    reconstructed = deployment.reconstruct(measurements)
    reconstructed_np = reconstructed.cpu().numpy().squeeze()
    
    # Save reconstructed image
    reconstructed_save_path = os.path.join(args.output_dir, 'reconstructed_image.png')
    reconstructed_uint8 = (np.clip(reconstructed_np, 0, 1) * 255).astype(np.uint8)
    Image.fromarray(reconstructed_uint8).save(reconstructed_save_path)
    print(f"  Saved reconstructed image to: {reconstructed_save_path}")
    
    # Visualization
    if original_image is not None:
        # We have original for comparison
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # Original
        axes[0].imshow(original_image.cpu().numpy().squeeze(), cmap='gray', vmin=0, vmax=1)
        axes[0].set_title(f'Original\n{os.path.basename(original_path)}')
        axes[0].axis('off')
        
        # Measurements
        axes[1].plot(measurements.cpu().numpy().squeeze())
        axes[1].set_title(f'1D Measurements (m={n_measurements})')
        axes[1].set_xlabel('Index')
        axes[1].set_ylabel('Intensity')
        axes[1].grid(True, alpha=0.3)
        
        # Reconstructed
        axes[2].imshow(reconstructed_np, cmap='gray', vmin=0, vmax=1)
        axes[2].set_title('Reconstructed')
        axes[2].axis('off')
        
        plt.tight_layout()
        
        # Compute metrics
        from losses import compute_psnr, SSIM
        psnr = compute_psnr(reconstructed, original_image)
        ssim_fn = SSIM().to(device)
        ssim = ssim_fn(reconstructed, original_image).item()
        
        fig.suptitle(f'PSNR: {psnr:.2f} dB | SSIM: {ssim:.4f}', fontsize=12, y=1.02)
        
        print(f"\nReconstruction metrics:")
        print(f"  PSNR: {psnr:.2f} dB")
        print(f"  SSIM: {ssim:.4f}")
    else:
        # No original, just show measurements and reconstruction
        fig, axes = plt.subplots(1, 2, figsize=(12, 5))
        
        # Measurements
        axes[0].plot(measurements.cpu().numpy().squeeze())
        axes[0].set_title(f'1D Measurements (m={n_measurements})')
        axes[0].set_xlabel('Index')
        axes[0].set_ylabel('Intensity')
        axes[0].grid(True, alpha=0.3)
        
        # Reconstructed
        axes[1].imshow(reconstructed_np, cmap='gray', vmin=0, vmax=1)
        axes[1].set_title('Reconstructed from measurements')
        axes[1].axis('off')
        
        plt.tight_layout()
    
    visualization_path = os.path.join(args.output_dir, 'reconstruction_result.png')
    plt.savefig(visualization_path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"  Saved visualization to: {visualization_path}")
    
    # Summary
    print("\n" + "="*60)
    print("Deployment complete!")
    print("="*60)
    print(f"  Output directory: {args.output_dir}")
    print(f"  Patterns: {args.patterns_file or os.path.join(args.output_dir, 'patterns.npy')}")
    print(f"  DMD patterns: {export_dir}")
    if original_image is not None:
        print(f"  Sampled measurements: {os.path.join(args.output_dir, 'sampled_measurements.npy')}")
    print(f"  Reconstructed image: {reconstructed_save_path}")
    print(f"  Visualization: {visualization_path}")


if __name__ == "__main__":
    main()
