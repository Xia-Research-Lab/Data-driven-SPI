"""
Deployment script with Pruning and Fine-tuning for SPI Generator

This script:
1. Loads a trained Generator model
2. Applies structured/unstructured pruning to reduce model size
3. Fine-tunes the pruned model to recover performance
4. Exports the pruned model for deployment

Usage:
    python deploy_pruned.py --checkpoint checkpoints/best_model.pth \
                            --prune_ratio 0.3 \
                            --finetune_epochs 10 \
                            --output_dir deploy_pruned_output
"""

import os
import argparse
import copy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.utils.prune as prune
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt

from model import Generator, FixedPatternEncoder
from losses import compute_psnr, SSIM


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='SPI Generator Pruning and Deployment')
    
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
    
    # Pruning config
    parser.add_argument('--prune_ratio', type=float, default=0.3,
                        help='Pruning ratio (0.0-1.0, higher = more pruning)')
    parser.add_argument('--prune_method', type=str, default='l1_unstructured',
                        choices=['l1_unstructured', 'ln_structured', 'random_unstructured'],
                        help='Pruning method')
    parser.add_argument('--prune_layers', type=str, default='all',
                        choices=['conv', 'linear', 'all'],
                        help='Which layers to prune (default: all, includes both conv and linear)')
    parser.add_argument('--save_sparse', action='store_true',
                        help='Save model in sparse format to reduce file size')
    
    # Fine-tuning config
    parser.add_argument('--finetune_epochs', type=int, default=10,
                        help='Number of fine-tuning epochs')
    parser.add_argument('--finetune_lr', type=float, default=1e-6,
                        help='Learning rate for fine-tuning')
    parser.add_argument('--batch_size', type=int, default=16,
                        help='Batch size for fine-tuning')
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    
    # Files
    parser.add_argument('--checkpoint', type=str, default='checkpoints/best_model.pth',
                        help='Path to model checkpoint')
    parser.add_argument('--data_dir', type=str, default='cyto128',
                        help='Path to dataset directory')
    parser.add_argument('--output_dir', type=str, default='pruned_for_deploy',
                        help='Directory to save outputs')
    
    # Noise config
    parser.add_argument('--noise_std', type=float, default=0.0,
                        help='Noise standard deviation for fine-tuning (0 = no noise)')
    
    return parser.parse_args()


class PrunedGeneratorDeployment:
    """
    Class for pruning, fine-tuning, and deploying the SPI Generator.
    """
    
    def __init__(self, n_measurements, img_size=128, base_features=64, 
                 pattern_type='hadamard', device='cuda'):
        self.n_measurements = n_measurements
        self.img_size = img_size
        self.n_pixels = img_size * img_size
        self.base_features = base_features
        self.pattern_type = pattern_type
        self.device = device
        
        # Initialize generator
        self.generator = Generator(
            n_measurements=n_measurements,
            n_pixels=self.n_pixels,
            img_size=img_size,
            base_features=base_features
        ).to(device)
        
        # Initialize encoder for simulation
        self.encoder = FixedPatternEncoder(
            img_size=img_size,
            n_measurements=n_measurements,
            pattern_type=pattern_type
        ).to(device)
        
        self.patterns = self.encoder.get_patterns()
        
        # Track pruning state
        self.is_pruned = False
        self.prune_info = {}
        
        print(f"PrunedGeneratorDeployment initialized:")
        print(f"  Image size: {img_size}x{img_size}")
        print(f"  Measurements: {n_measurements}")
        print(f"  Pattern type: {pattern_type}")
        print(f"  Generator parameters: {self._count_parameters():,}")
    
    def _count_parameters(self, count_zeros=True):
        """Count total and non-zero parameters."""
        total = 0
        nonzero = 0
        for p in self.generator.parameters():
            total += p.numel()
            if count_zeros:
                nonzero += (p != 0).sum().item()
        return total if count_zeros else nonzero
    
    def _count_nonzero_parameters(self):
        """Count non-zero parameters (considering pruning masks)."""
        nonzero = 0
        for name, module in self.generator.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                # Check if pruning mask exists
                if hasattr(module, 'weight_mask'):
                    # Pruned layer - count non-zero in masked weight
                    masked_weight = module.weight_orig * module.weight_mask
                    nonzero += (masked_weight != 0).sum().item()
                else:
                    # Not pruned - count directly
                    nonzero += (module.weight != 0).sum().item()
                # Count bias if exists
                if module.bias is not None:
                    nonzero += (module.bias != 0).sum().item()
            elif isinstance(module, nn.BatchNorm2d):
                if module.weight is not None:
                    nonzero += (module.weight != 0).sum().item()
                if module.bias is not None:
                    nonzero += (module.bias != 0).sum().item()
        return nonzero
    
    def _count_total_prunable_parameters(self):
        """Count total parameters in prunable layers."""
        total = 0
        for name, module in self.generator.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                total += module.weight.numel()
        return total
    
    def load_generator_weights(self, checkpoint_path):
        """Load generator weights from checkpoint."""
        if not os.path.exists(checkpoint_path):
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location=self.device)
        
        # Handle different checkpoint formats
        state_dict = checkpoint.get('generator_state_dict', 
                                   checkpoint.get('model_state_dict', checkpoint))
        
        # Extract generator weights
        generator_state_dict = {}
        for key, value in state_dict.items():
            if key.startswith('generator.'):
                new_key = key[len('generator.'):]
                generator_state_dict[new_key] = value
            elif not key.startswith('encoder.') and not key.startswith('discriminator.'):
                generator_state_dict[key] = value
        
        if generator_state_dict:
            self.generator.load_state_dict(generator_state_dict, strict=False)
        else:
            self.generator.load_state_dict(state_dict, strict=False)
        
        print(f"Loaded generator weights from {checkpoint_path}")
        print(f"  Checkpoint epoch: {checkpoint.get('epoch', 'unknown')}")
    
    def get_prunable_layers(self, layer_type='conv'):
        """Get list of layers to prune."""
        layers = []
        
        for name, module in self.generator.named_modules():
            if layer_type == 'conv' or layer_type == 'all':
                if isinstance(module, nn.Conv2d):
                    layers.append((module, 'weight'))
            if layer_type == 'linear' or layer_type == 'all':
                if isinstance(module, nn.Linear):
                    layers.append((module, 'weight'))
        
        return layers
    
    def apply_pruning(self, prune_ratio=0.3, method='l1_unstructured', layer_type='conv'):
        """
        Apply pruning to the generator.
        
        Args:
            prune_ratio: Fraction of weights to prune (0.0 to 1.0)
            method: Pruning method ('l1_unstructured', 'ln_structured', 'random_unstructured')
            layer_type: Which layers to prune ('conv', 'linear', 'all')
        """
        print(f"\n{'='*60}")
        print(f"Applying Pruning")
        print(f"{'='*60}")
        print(f"  Method: {method}")
        print(f"  Prune ratio: {prune_ratio:.1%}")
        print(f"  Layer type: {layer_type}")
        
        layers = self.get_prunable_layers(layer_type)
        print(f"  Prunable layers: {len(layers)}")
        
        # Count parameters before pruning
        params_before = self._count_parameters()
        nonzero_before = self._count_nonzero_parameters()
        
        # Apply pruning based on method
        if method == 'l1_unstructured':
            # Global L1 unstructured pruning
            prune.global_unstructured(
                layers,
                pruning_method=prune.L1Unstructured,
                amount=prune_ratio,
            )
        elif method == 'ln_structured':
            # Per-layer structured pruning (prune entire filters)
            for module, name in layers:
                if isinstance(module, nn.Conv2d):
                    prune.ln_structured(module, name=name, amount=prune_ratio, n=2, dim=0)
        elif method == 'random_unstructured':
            # Random unstructured pruning
            prune.global_unstructured(
                layers,
                pruning_method=prune.RandomUnstructured,
                amount=prune_ratio,
            )
        else:
            raise ValueError(f"Unknown pruning method: {method}")
        
        # Count parameters after pruning
        nonzero_after = self._count_nonzero_parameters()
        actual_prune_ratio = 1 - (nonzero_after / nonzero_before)
        
        self.is_pruned = True
        self.prune_info = {
            'method': method,
            'layer_type': layer_type,
            'target_ratio': prune_ratio,
            'actual_ratio': actual_prune_ratio,
            'params_total': params_before,
            'params_nonzero_before': nonzero_before,
            'params_nonzero_after': nonzero_after,
            'layers_pruned': len(layers),
        }
        
        print(f"\nPruning Results:")
        print(f"  Total parameters: {params_before:,}")
        print(f"  Non-zero before: {nonzero_before:,}")
        print(f"  Non-zero after: {nonzero_after:,}")
        print(f"  Actual sparsity: {actual_prune_ratio:.2%}")
        print(f"  Size reduction (theoretical): {actual_prune_ratio:.2%}")
        
        return self.prune_info
    
    def remove_pruning_reparametrization(self):
        """
        Remove pruning reparametrization to make pruning permanent.
        This converts the pruned model to a regular model with zeros.
        """
        for name, module in self.generator.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                try:
                    prune.remove(module, 'weight')
                except ValueError:
                    pass  # No pruning was applied to this layer
        
        print("Removed pruning reparametrization (pruning is now permanent)")
    
    def simulate_measurement(self, images, noise_std=0.0):
        """Simulate measurements with optional noise."""
        batch_size = images.shape[0]
        images = images.to(self.device)
        
        # Flatten and compute measurements
        x_flat = images.view(batch_size, -1)
        measurements = F.linear(x_flat, self.patterns)
        
        # Add noise if specified
        if noise_std > 0:
            noise = torch.randn_like(measurements) * noise_std
            measurements = measurements + noise
        
        return measurements
    
    @torch.no_grad()
    def evaluate(self, dataloader, noise_std=0.0):
        """Evaluate the generator on a dataset."""
        self.generator.eval()
        
        total_psnr = 0
        total_ssim = 0
        n_batches = 0
        
        ssim_fn = SSIM().to(self.device)
        
        for images in dataloader:
            if isinstance(images, (list, tuple)):
                images = images[0]
            images = images.to(self.device)
            
            # Simulate measurements
            measurements = self.simulate_measurement(images, noise_std)
            
            # Reconstruct
            reconstructed = self.generator(measurements)
            
            # Compute metrics
            psnr = compute_psnr(reconstructed, images)
            ssim = ssim_fn(reconstructed, images).item()
            
            total_psnr += psnr
            total_ssim += ssim
            n_batches += 1
        
        avg_psnr = total_psnr / n_batches
        avg_ssim = total_ssim / n_batches
        
        return {'psnr': avg_psnr, 'ssim': avg_ssim}
    
    def finetune(self, train_loader, val_loader, epochs=10, lr=1e-4, noise_std=0.0):
        """
        Fine-tune the pruned generator.
        
        Args:
            train_loader: Training data loader
            val_loader: Validation data loader
            epochs: Number of fine-tuning epochs
            lr: Learning rate
            noise_std: Noise standard deviation
        """
        print(f"\n{'='*60}")
        print(f"Fine-tuning Pruned Generator")
        print(f"{'='*60}")
        print(f"  Epochs: {epochs}")
        print(f"  Learning rate: {lr}")
        print(f"  Noise std: {noise_std}")
        
        # Optimizer - only optimize non-pruned weights
        optimizer = torch.optim.Adam(self.generator.parameters(), lr=lr)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
        
        criterion = nn.MSELoss()
        ssim_fn = SSIM().to(self.device)
        
        # Evaluate before fine-tuning
        print("\nBefore fine-tuning:")
        metrics_before = self.evaluate(val_loader, noise_std)
        print(f"  Val PSNR: {metrics_before['psnr']:.2f} dB")
        print(f"  Val SSIM: {metrics_before['ssim']:.4f}")
        
        best_psnr = metrics_before['psnr']
        best_state = copy.deepcopy(self.generator.state_dict())
        
        # Training loop
        for epoch in range(epochs):
            self.generator.train()
            epoch_loss = 0
            epoch_psnr = 0
            
            pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{epochs}")
            for images in pbar:
                if isinstance(images, (list, tuple)):
                    images = images[0]
                images = images.to(self.device)
                
                # Simulate measurements
                measurements = self.simulate_measurement(images, noise_std)
                
                # Forward pass
                reconstructed = self.generator(measurements)
                
                # Loss
                loss = criterion(reconstructed, images)
                
                # Backward pass
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                # Metrics
                with torch.no_grad():
                    psnr = compute_psnr(reconstructed, images)
                
                epoch_loss += loss.item()
                epoch_psnr += psnr
                
                pbar.set_postfix(loss=f"{loss.item():.4f}", psnr=f"{psnr:.2f}")
            
            scheduler.step()
            
            # Validation
            val_metrics = self.evaluate(val_loader, noise_std)
            
            avg_loss = epoch_loss / len(train_loader)
            avg_psnr = epoch_psnr / len(train_loader)
            
            print(f"\nEpoch {epoch+1}/{epochs}:")
            print(f"  Train Loss: {avg_loss:.4f}")
            print(f"  Train PSNR: {avg_psnr:.2f} dB")
            print(f"  Val PSNR: {val_metrics['psnr']:.2f} dB")
            print(f"  Val SSIM: {val_metrics['ssim']:.4f}")
            
            # Save best model
            if val_metrics['psnr'] > best_psnr:
                best_psnr = val_metrics['psnr']
                best_state = copy.deepcopy(self.generator.state_dict())
                print(f"  *** New best PSNR: {best_psnr:.2f} dB ***")
        
        # Load best model
        self.generator.load_state_dict(best_state)
        
        # Evaluate after fine-tuning
        print("\nAfter fine-tuning:")
        metrics_after = self.evaluate(val_loader, noise_std)
        print(f"  Val PSNR: {metrics_after['psnr']:.2f} dB")
        print(f"  Val SSIM: {metrics_after['ssim']:.4f}")
        print(f"  PSNR improvement: {metrics_after['psnr'] - metrics_before['psnr']:.2f} dB")
        
        return {
            'before': metrics_before,
            'after': metrics_after,
            'best_psnr': best_psnr
        }
    
    def export_model(self, output_dir, model_name='generator_pruned', save_sparse=False):
        """
        Export the pruned model for deployment.
        
        Creates:
        - generator_pruned.pth: PyTorch state dict (dense format)
        - generator_pruned_sparse.pth: Sparse format (if save_sparse=True)
        - generator_pruned_full.pth: Full checkpoint with metadata
        - generator_pruned.onnx: ONNX format (optional)
        - model_info.txt: Model information
        
        Args:
            output_dir: Output directory
            model_name: Base name for output files
            save_sparse: Whether to save in sparse format for smaller file size
        """
        os.makedirs(output_dir, exist_ok=True)
        
        # Remove pruning reparametrization to make it permanent
        self.remove_pruning_reparametrization()
        
        # 1. Save state dict (dense format)
        state_dict_path = os.path.join(output_dir, f'{model_name}.pth')
        torch.save(self.generator.state_dict(), state_dict_path)
        print(f"Saved state dict to: {state_dict_path}")
        dense_size = os.path.getsize(state_dict_path) / (1024 * 1024)
        
        # 2. Save compressed format (using gzip) if requested
        sparse_size = None
        if save_sparse:
            import gzip
            import pickle
            
            # Convert to half precision for smaller size
            compressed_state_dict = {}
            for key, tensor in self.generator.state_dict().items():
                if tensor.dtype == torch.float32:
                    compressed_state_dict[key] = tensor.half()
                else:
                    compressed_state_dict[key] = tensor
            
            compressed_path = os.path.join(output_dir, f'{model_name}_compressed.pth.gz')
            with gzip.open(compressed_path, 'wb') as f:
                pickle.dump(compressed_state_dict, f)
            sparse_size = os.path.getsize(compressed_path) / (1024 * 1024)
            print(f"Saved compressed model to: {compressed_path}")
            print(f"  Dense size (FP32): {dense_size:.2f} MB")
            print(f"  Compressed size (FP16+gzip): {sparse_size:.2f} MB")
            print(f"  Compression ratio: {dense_size/sparse_size:.2f}x")
        
        # 3. Save full checkpoint with metadata
        full_checkpoint = {
            'generator_state_dict': self.generator.state_dict(),
            'model_config': {
                'n_measurements': self.n_measurements,
                'n_pixels': self.n_pixels,
                'img_size': self.img_size,
                'base_features': self.base_features,
            },
            'prune_info': self.prune_info,
            'pattern_type': self.pattern_type,
        }
        full_path = os.path.join(output_dir, f'{model_name}_full.pth')
        torch.save(full_checkpoint, full_path)
        print(f"Saved full checkpoint to: {full_path}")
        
        # 4. Save patterns
        patterns_path = os.path.join(output_dir, 'patterns.npy')
        np.save(patterns_path, self.patterns.cpu().numpy())
        print(f"Saved patterns to: {patterns_path}")
        
        # 5. Try to export ONNX (optional)
        try:
            onnx_path = os.path.join(output_dir, f'{model_name}.onnx')
            dummy_input = torch.randn(1, self.n_measurements).to(self.device)
            self.generator.eval()
            torch.onnx.export(
                self.generator,
                dummy_input,
                onnx_path,
                input_names=['measurements'],
                output_names=['reconstructed'],
                dynamic_axes={
                    'measurements': {0: 'batch_size'},
                    'reconstructed': {0: 'batch_size'}
                },
                opset_version=11
            )
            print(f"Saved ONNX model to: {onnx_path}")
        except Exception as e:
            print(f"Could not export ONNX model: {e}")
        
        # 6. Save model info
        info_path = os.path.join(output_dir, 'model_info.txt')
        with open(info_path, 'w') as f:
            f.write("SPI Generator - Pruned Model Information\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Model Configuration:\n")
            f.write(f"  Image size: {self.img_size}x{self.img_size}\n")
            f.write(f"  Measurements: {self.n_measurements}\n")
            f.write(f"  Base features: {self.base_features}\n")
            f.write(f"  Pattern type: {self.pattern_type}\n\n")
            
            if self.prune_info:
                f.write(f"Pruning Information:\n")
                f.write(f"  Method: {self.prune_info.get('method', 'unknown')}\n")
                f.write(f"  Layers pruned: {self.prune_info.get('layer_type', 'unknown')}\n")
                f.write(f"  Target ratio: {self.prune_info.get('target_ratio', 0):.2%}\n")
                f.write(f"  Actual sparsity: {self.prune_info.get('actual_ratio', 0):.2%}\n")
                f.write(f"  Total parameters: {self.prune_info.get('params_total', 0):,}\n")
                f.write(f"  Non-zero after pruning: {self.prune_info.get('params_nonzero_after', 0):,}\n")
                f.write(f"\nFile Sizes:\n")
                f.write(f"  Dense model (FP32): {dense_size:.2f} MB\n")
                if sparse_size:
                    f.write(f"  Compressed model (FP16+gzip): {sparse_size:.2f} MB\n")
                    f.write(f"  Compression: {dense_size/sparse_size:.2f}x\n")
            
            f.write(f"\nFiles:\n")
            f.write(f"  - {model_name}.pth: State dict (FP32, dense)\n")
            if save_sparse:
                f.write(f"  - {model_name}_compressed.pth.gz: State dict (FP16, gzip compressed)\n")
            f.write(f"  - {model_name}_full.pth: Full checkpoint with metadata\n")
            f.write(f"  - patterns.npy: Measurement patterns\n")
        
        print(f"Saved model info to: {info_path}")
        
        # Calculate model size
        print(f"\nExported model size: {dense_size:.2f} MB")
        
        return {
            'state_dict_path': state_dict_path,
            'full_path': full_path,
            'patterns_path': patterns_path,
            'info_path': info_path,
            'size_mb': dense_size,
            'sparse_size_mb': sparse_size
        }
    
    def _convert_to_sparse(self, state_dict):
        """
        Convert dense state dict to compressed format for smaller file size.
        Stores only non-zero values and their indices.
        """
        sparse_dict = {}
        total_original = 0
        total_compressed = 0
        
        for key, tensor in state_dict.items():
            tensor_np = tensor.cpu().numpy()
            original_size = tensor_np.nbytes
            total_original += original_size
            
            # Only convert tensors with >30% sparsity and large enough
            sparsity = (tensor_np == 0).mean()
            if sparsity > 0.3 and tensor_np.size > 10000:
                # Find non-zero elements
                flat = tensor_np.flatten()
                nonzero_mask = flat != 0
                nonzero_indices = np.where(nonzero_mask)[0].astype(np.int32)
                nonzero_values = flat[nonzero_mask].astype(np.float32)
                
                sparse_dict[key] = {
                    'sparse': True,
                    'indices': nonzero_indices,
                    'values': nonzero_values,
                    'shape': tensor_np.shape,
                    'dtype': str(tensor_np.dtype),
                }
                compressed_size = nonzero_indices.nbytes + nonzero_values.nbytes
                total_compressed += compressed_size
            else:
                # Keep as dense
                sparse_dict[key] = tensor
                total_compressed += original_size
        
        print(f"  Sparse conversion: {total_original/1e6:.1f} MB -> {total_compressed/1e6:.1f} MB")
        return sparse_dict
    
    @staticmethod
    def load_sparse_state_dict(sparse_path, device='cpu'):
        """
        Load a sparse state dict and convert back to dense tensors.
        
        Args:
            sparse_path: Path to sparse checkpoint
            device: Device to load tensors to
        
        Returns:
            Dense state dict
        """
        sparse_dict = torch.load(sparse_path, map_location='cpu')
        dense_dict = {}
        
        for key, value in sparse_dict.items():
            if isinstance(value, dict) and value.get('sparse', False):
                # Reconstruct dense tensor
                shape = value['shape']
                flat_size = np.prod(shape)
                flat = np.zeros(flat_size, dtype=np.float32)
                flat[value['indices']] = value['values']
                tensor = torch.from_numpy(flat.reshape(shape)).to(device)
                dense_dict[key] = tensor
            else:
                dense_dict[key] = value.to(device) if isinstance(value, torch.Tensor) else value
        
        return dense_dict
    
    def visualize_pruning(self, output_path):
        """Visualize weight distribution before and after pruning."""
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Collect all weights
        all_weights = []
        layer_names = []
        
        for name, module in self.generator.named_modules():
            if isinstance(module, (nn.Conv2d, nn.Linear)):
                weights = module.weight.data.cpu().numpy().flatten()
                all_weights.append(weights)
                layer_names.append(name)
        
        # Plot 1: Overall weight distribution
        all_weights_flat = np.concatenate(all_weights)
        axes[0, 0].hist(all_weights_flat, bins=100, density=True, alpha=0.7)
        axes[0, 0].set_title('Weight Distribution (All Layers)')
        axes[0, 0].set_xlabel('Weight Value')
        axes[0, 0].set_ylabel('Density')
        axes[0, 0].axvline(x=0, color='r', linestyle='--', alpha=0.5)
        
        # Plot 2: Sparsity per layer
        sparsities = []
        for weights in all_weights:
            sparsity = (weights == 0).mean()
            sparsities.append(sparsity)
        
        axes[0, 1].barh(range(len(layer_names)), sparsities)
        axes[0, 1].set_yticks(range(len(layer_names)))
        axes[0, 1].set_yticklabels([n[-20:] for n in layer_names], fontsize=6)
        axes[0, 1].set_xlabel('Sparsity')
        axes[0, 1].set_title('Sparsity per Layer')
        
        # Plot 3: Weight magnitude distribution (non-zero only)
        nonzero_weights = all_weights_flat[all_weights_flat != 0]
        if len(nonzero_weights) > 0:
            axes[1, 0].hist(np.abs(nonzero_weights), bins=100, density=True, alpha=0.7)
            axes[1, 0].set_title('Non-zero Weight Magnitudes')
            axes[1, 0].set_xlabel('|Weight|')
            axes[1, 0].set_ylabel('Density')
        
        # Plot 4: Statistics summary
        axes[1, 1].axis('off')
        total_params = len(all_weights_flat)
        zero_params = (all_weights_flat == 0).sum()
        nonzero_params = total_params - zero_params
        
        summary_text = f"""
        Pruning Summary
        ===============
        Total parameters: {total_params:,}
        Zero parameters: {zero_params:,}
        Non-zero parameters: {nonzero_params:,}
        Overall sparsity: {zero_params/total_params:.2%}
        
        Weight Statistics (non-zero):
        Mean: {np.mean(nonzero_weights):.6f}
        Std: {np.std(nonzero_weights):.6f}
        Min: {np.min(nonzero_weights):.6f}
        Max: {np.max(nonzero_weights):.6f}
        """
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=10, family='monospace',
                       verticalalignment='center')
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close()
        print(f"Saved pruning visualization to: {output_path}")


def get_data_loaders(data_dir, img_size, batch_size, num_workers):
    """Create data loaders for fine-tuning."""
    from torchvision import transforms
    from torch.utils.data import DataLoader, Dataset
    
    transform = transforms.Compose([
        transforms.Grayscale(num_output_channels=1),
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
    ])
    
    class FlatImageDataset(Dataset):
        """Dataset for flat directory structure (images directly in folder)."""
        def __init__(self, root_dir, transform=None):
            self.root_dir = root_dir
            self.transform = transform
            self.image_files = sorted([
                f for f in os.listdir(root_dir)
                if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
            ])
        
        def __len__(self):
            return len(self.image_files)
        
        def __getitem__(self, idx):
            img_path = os.path.join(self.root_dir, self.image_files[idx])
            image = Image.open(img_path).convert('RGB')
            if self.transform:
                image = self.transform(image)
            return image
    
    # Check directory structure
    train_dir = os.path.join(data_dir, 'train')
    val_dir = os.path.join(data_dir, 'valid')
    
    if os.path.exists(train_dir):
        # Has train/valid subdirectories - use FlatImageDataset
        train_dataset = FlatImageDataset(train_dir, transform=transform)
        if os.path.exists(val_dir):
            val_dataset = FlatImageDataset(val_dir, transform=transform)
        else:
            val_dataset = train_dataset
    else:
        # Flat structure - use the main directory
        train_dataset = FlatImageDataset(data_dir, transform=transform)
        val_dataset = train_dataset
    
    train_loader = DataLoader(
        train_dataset, 
        batch_size=batch_size, 
        shuffle=True, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=batch_size, 
        shuffle=False, 
        num_workers=num_workers,
        pin_memory=True
    )
    
    print(f"Data loaders created:")
    print(f"  Train samples: {len(train_dataset)}")
    print(f"  Val samples: {len(val_dataset)}")
    
    return train_loader, val_loader


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
        raise FileNotFoundError(f"Checkpoint not found: {args.checkpoint}")
    
    # Create deployment object
    deployment = PrunedGeneratorDeployment(
        n_measurements=n_measurements,
        img_size=args.img_size,
        base_features=base_features,
        pattern_type=pattern_type,
        device=device
    )
    
    # Load generator weights
    deployment.load_generator_weights(args.checkpoint)
    
    # Get data loaders for fine-tuning
    train_loader, val_loader = get_data_loaders(
        args.data_dir, args.img_size, args.batch_size, args.num_workers
    )
    
    # Evaluate before pruning
    print(f"\n{'='*60}")
    print("Evaluation Before Pruning")
    print(f"{'='*60}")
    metrics_before = deployment.evaluate(val_loader, args.noise_std)
    print(f"  PSNR: {metrics_before['psnr']:.2f} dB")
    print(f"  SSIM: {metrics_before['ssim']:.4f}")
    
    # Apply pruning
    deployment.apply_pruning(
        prune_ratio=args.prune_ratio,
        method=args.prune_method,
        layer_type=args.prune_layers
    )
    
    # Evaluate after pruning (before fine-tuning)
    print(f"\n{'='*60}")
    print("Evaluation After Pruning (Before Fine-tuning)")
    print(f"{'='*60}")
    metrics_after_prune = deployment.evaluate(val_loader, args.noise_std)
    print(f"  PSNR: {metrics_after_prune['psnr']:.2f} dB")
    print(f"  SSIM: {metrics_after_prune['ssim']:.4f}")
    print(f"  PSNR drop: {metrics_before['psnr'] - metrics_after_prune['psnr']:.2f} dB")
    
    # Fine-tune
    finetune_results = None
    if args.finetune_epochs > 0:
        finetune_results = deployment.finetune(
            train_loader, 
            val_loader, 
            epochs=args.finetune_epochs,
            lr=args.finetune_lr,
            noise_std=args.noise_std
        )
    
    # Visualize pruning
    viz_path = os.path.join(args.output_dir, 'pruning_visualization.png')
    deployment.visualize_pruning(viz_path)
    
    # Export model
    print(f"\n{'='*60}")
    print("Exporting Pruned Model")
    print(f"{'='*60}")
    export_info = deployment.export_model(
        args.output_dir, 
        save_sparse=args.save_sparse
    )
    
    # Final summary
    print(f"\n{'='*60}")
    print("Pruning and Deployment Complete!")
    print(f"{'='*60}")
    print(f"  Output directory: {args.output_dir}")
    print(f"  Pruned layers: {args.prune_layers}")
    print(f"  Original PSNR: {metrics_before['psnr']:.2f} dB")
    print(f"  After pruning: {metrics_after_prune['psnr']:.2f} dB")
    if finetune_results:
        print(f"  After fine-tuning: {finetune_results['after']['psnr']:.2f} dB")
    print(f"  Model size (dense): {export_info['size_mb']:.2f} MB")
    if export_info.get('sparse_size_mb'):
        print(f"  Model size (sparse): {export_info['sparse_size_mb']:.2f} MB")
    print(f"  Sparsity: {deployment.prune_info['actual_ratio']:.2%}")


if __name__ == "__main__":
    main()
