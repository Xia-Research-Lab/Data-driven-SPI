"""
Utilities for LED (Learned Encoder-Decoder) Single-Pixel Imaging
Dataset loaders and helper functions
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np


class Mish(nn.Module):
    """
    Mish activation function: x * tanh(softplus(x))
    Reference: https://arxiv.org/abs/1908.08681
    """
    def forward(self, x):
        return x * torch.tanh(F.softplus(x))


def mish(x):
    """Functional version of Mish activation"""
    return x * torch.tanh(F.softplus(x))


class CytoImageDataset(Dataset):
    """
    Dataset for loading grayscale images from CytoImageNet-style folder structure.
    
    Args:
        root_dir: Path to the dataset folder (e.g., 'cyto128/train')
        img_size: Target image size (default: 128)
        transform: Optional additional transforms
    """
    def __init__(self, root_dir, img_size=128, transform=None):
        self.root_dir = root_dir
        self.img_size = img_size
        self.transform = transform
        
        # Get all image files
        self.image_files = sorted([
            f for f in os.listdir(root_dir) 
            if f.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff'))
        ])
        
        # Default transform: resize to img_size, convert to grayscale, normalize to [0, 1]
        self.default_transform = transforms.Compose([
            transforms.Grayscale(num_output_channels=1),
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),  # Converts to [0, 1] range
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        img_path = os.path.join(self.root_dir, self.image_files[idx])
        image = Image.open(img_path).convert('RGB')  # Load as RGB first
        
        # Apply default transform
        image = self.default_transform(image)
        
        # Apply additional transforms if provided
        if self.transform is not None:
            image = self.transform(image)
        
        return image


def get_dataloaders(data_dir, img_size=128, batch_size=32, num_workers=4):
    """
    Create train, validation, and test dataloaders.
    
    Args:
        data_dir: Root directory containing train/, valid/, test/ subdirectories
        img_size: Target image size
        batch_size: Batch size for dataloaders
        num_workers: Number of workers for data loading
    
    Returns:
        dict: Dictionary containing 'train', 'valid', 'test' dataloaders
    """
    dataloaders = {}
    
    for split in ['train', 'valid', 'test']:
        split_dir = os.path.join(data_dir, split)
        if os.path.exists(split_dir):
            dataset = CytoImageDataset(split_dir, img_size=img_size)
            shuffle = (split == 'train')
            dataloaders[split] = DataLoader(
                dataset, 
                batch_size=batch_size, 
                shuffle=shuffle,
                num_workers=num_workers,
                pin_memory=True,
                drop_last=(split == 'train')
            )
            print(f"Loaded {split} dataset: {len(dataset)} images")
    
    return dataloaders


def add_gaussian_noise(measurements, noise_std):
    """
    Add Gaussian noise to measurements.
    
    Args:
        measurements: Input tensor of shape (batch, m)
        noise_std: Standard deviation of Gaussian noise
    
    Returns:
        Noisy measurements
    """
    if noise_std > 0:
        noise = torch.randn_like(measurements) * noise_std
        return measurements + noise
    return measurements


def compute_psnr(img1, img2, max_val=1.0):
    """
    Compute Peak Signal-to-Noise Ratio between two images.
    
    Args:
        img1, img2: Input images (tensors)
        max_val: Maximum possible pixel value
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(img1, img2)
    if mse == 0:
        return float('inf')
    psnr = 20 * torch.log10(max_val / torch.sqrt(mse))
    return psnr.item()


def count_parameters(model):
    """Count the number of trainable parameters in a model."""
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_checkpoint(model, optimizer, epoch, loss, filepath):
    """Save model checkpoint."""
    checkpoint = {
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'loss': loss,
    }
    torch.save(checkpoint, filepath)
    print(f"Checkpoint saved to {filepath}")


def load_checkpoint(model, optimizer, filepath):
    """Load model checkpoint."""
    checkpoint = torch.load(filepath)
    model.load_state_dict(checkpoint['model_state_dict'])
    if optimizer is not None:
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    epoch = checkpoint['epoch']
    loss = checkpoint['loss']
    print(f"Checkpoint loaded from {filepath}, epoch {epoch}")
    return epoch, loss


class EarlyStopping:
    """
    Early stopping utility to stop training when validation loss stops improving.
    """
    def __init__(self, patience=10, min_delta=0, mode='min'):
        self.patience = patience
        self.min_delta = min_delta
        self.mode = mode
        self.counter = 0
        self.best_value = None
        self.early_stop = False
    
    def __call__(self, value):
        if self.best_value is None:
            self.best_value = value
            return False
        
        if self.mode == 'min':
            improved = value < self.best_value - self.min_delta
        else:
            improved = value > self.best_value + self.min_delta
        
        if improved:
            self.best_value = value
            self.counter = 0
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        
        return self.early_stop


def visualize_masks(encoder_weight, n_masks=8, img_size=128):
    """
    Visualize learned measurement masks.
    
    Args:
        encoder_weight: Encoder weight matrix of shape (m, n)
        n_masks: Number of masks to visualize
        img_size: Image size for reshaping masks
    
    Returns:
        numpy array of mask visualizations
    """
    import matplotlib.pyplot as plt
    
    m = encoder_weight.shape[0]
    n_masks = min(n_masks, m)
    
    fig, axes = plt.subplots(1, n_masks, figsize=(2*n_masks, 2))
    
    for i in range(n_masks):
        mask = encoder_weight[i].detach().cpu().numpy().reshape(img_size, img_size)
        axes[i].imshow(mask, cmap='gray')
        axes[i].axis('off')
        axes[i].set_title(f'Mask {i+1}')
    
    plt.tight_layout()
    return fig


if __name__ == "__main__":
    # Test the dataset loader
    print("Testing CytoImageDataset...")
    dataset = CytoImageDataset("cyto128/train", img_size=128)
    print(f"Dataset size: {len(dataset)}")
    
    sample = dataset[0]
    print(f"Sample shape: {sample.shape}")
    print(f"Sample min: {sample.min():.4f}, max: {sample.max():.4f}")
    
    # Test dataloader
    print("\nTesting dataloaders...")
    loaders = get_dataloaders("cyto128", img_size=128, batch_size=16, num_workers=0)
    
    for split, loader in loaders.items():
        batch = next(iter(loader))
        print(f"{split}: batch shape = {batch.shape}")
