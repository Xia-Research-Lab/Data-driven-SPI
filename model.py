"""
SPI (Single-Pixel Imaging) Model with Fixed Patterns and GAN-based Reconstruction

Key Changes from LED:
1. Fixed measurement patterns (Hadamard zigzag or Random binary)
2. Decoder-only trainable network (Generator)
3. Added Discriminator for GAN training
4. Fixed 1024 measurements with -1/+1 binarization
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from utils import Mish


def generate_hadamard_matrix(n):
    """
    Generate Hadamard matrix of size n x n using Sylvester's construction.
    n must be a power of 2.
    
    Returns:
        Hadamard matrix with values -1 and +1
    """
    if n == 1:
        return np.array([[1]])
    
    # Find the largest power of 2 <= n
    k = int(np.log2(n))
    size = 2 ** k
    
    H = np.array([[1]])
    for _ in range(k):
        H = np.block([[H, H], [H, -H]])
    
    return H


def zigzag_order(n):
    """
    Generate zigzag ordering indices for an n x n matrix.
    
    Returns:
        Array of indices in zigzag order
    """
    indices = []
    for s in range(2 * n - 1):
        if s % 2 == 0:
            # Even diagonal: go down-left
            for i in range(min(s, n - 1), max(0, s - n + 1) - 1, -1):
                j = s - i
                indices.append(i * n + j)
        else:
            # Odd diagonal: go up-right
            for i in range(max(0, s - n + 1), min(s, n - 1) + 1):
                j = s - i
                indices.append(i * n + j)
    return np.array(indices)


def generate_hadamard_zigzag_patterns(img_size, n_measurements):
    """
    Generate Hadamard patterns reordered by zigzag (low-frequency first).
    
    Args:
        img_size: Image size (H = W)
        n_measurements: Number of measurements to select
    
    Returns:
        Pattern matrix of shape (n_measurements, img_size * img_size) with values -1/+1
    """
    n_pixels = img_size * img_size
    
    # Find the smallest power of 2 >= n_pixels
    hadamard_size = 2 ** int(np.ceil(np.log2(n_pixels)))
    
    # Generate full Hadamard matrix
    H = generate_hadamard_matrix(hadamard_size)
    
    # Truncate to n_pixels
    H = H[:n_pixels, :n_pixels]
    
    # Apply zigzag reordering to select low-frequency patterns first
    zigzag_idx = zigzag_order(img_size)
    
    # Reorder rows according to zigzag pattern
    H_zigzag = H[zigzag_idx, :]
    
    # Select first n_measurements patterns
    patterns = H_zigzag[:n_measurements, :].astype(np.float32)
    
    return patterns


def generate_random_binary_patterns(img_size, n_measurements, seed=42):
    """
    Generate random binary patterns with values -1/+1.
    
    Args:
        img_size: Image size (H = W)
        n_measurements: Number of measurements
        seed: Random seed for reproducibility
    
    Returns:
        Pattern matrix of shape (n_measurements, img_size * img_size) with values -1/+1
    """
    np.random.seed(seed)
    n_pixels = img_size * img_size
    
    # Generate random 0/1 matrix and convert to -1/+1
    patterns = np.random.randint(0, 2, size=(n_measurements, n_pixels)).astype(np.float32)
    patterns = patterns * 2 - 1  # Convert 0/1 to -1/+1
    
    return patterns


class FixedPatternEncoder(nn.Module):
    """
    Fixed pattern encoder for single-pixel imaging.
    
    Uses pre-generated -1/+1 binary patterns (not learnable).
    Supports Hadamard zigzag and Random binary patterns.
    
    Args:
        img_size: Image size (H = W, default: 128)
        n_measurements: Number of measurements (default: 1024)
        pattern_type: Type of patterns ('hadamard' or 'random')
        seed: Random seed for random patterns
    """
    def __init__(self, img_size=128, n_measurements=1024, pattern_type='hadamard', seed=42):
        super(FixedPatternEncoder, self).__init__()
        
        self.img_size = img_size
        self.n_pixels = img_size * img_size
        self.n_measurements = n_measurements
        self.pattern_type = pattern_type
        
        # Generate patterns
        if pattern_type == 'hadamard':
            patterns = generate_hadamard_zigzag_patterns(img_size, n_measurements)
        elif pattern_type == 'random':
            patterns = generate_random_binary_patterns(img_size, n_measurements, seed)
        else:
            raise ValueError(f"Unknown pattern type: {pattern_type}. Use 'hadamard' or 'random'.")
        
        # Register as buffer (not trainable, but moves with model to device)
        self.register_buffer('patterns', torch.from_numpy(patterns))
        
        print(f"FixedPatternEncoder initialized:")
        print(f"  Pattern type: {pattern_type}")
        print(f"  Shape: {self.patterns.shape}")
        print(f"  Values: {torch.unique(self.patterns).cpu().numpy()}")
    
    def forward(self, x):
        """
        Compute measurements using fixed patterns.
        
        Args:
            x: Input image batch of shape (B, 1, H, W)
        
        Returns:
            measurements: Tensor of shape (B, m)
        """
        batch_size = x.shape[0]
        # Flatten image: (B, 1, H, W) -> (B, n)
        x_flat = x.view(batch_size, -1)
        # Compute measurements: y = E @ x, using F.linear with patterns
        measurements = F.linear(x_flat, self.patterns)
        return measurements
    
    def get_patterns(self):
        """Return the fixed pattern matrix."""
        return self.patterns


class DoubleConv(nn.Module):
    """Double convolution block for U-Net."""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super(DoubleConv, self).__init__()
        if mid_channels is None:
            mid_channels = out_channels
        
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            Mish(),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            Mish()
        )
    
    def forward(self, x):
        return self.double_conv(x)


class Down(nn.Module):
    """Downsampling block for U-Net encoder."""
    def __init__(self, in_channels, out_channels):
        super(Down, self).__init__()
        self.maxpool_conv = nn.Sequential(
            nn.MaxPool2d(2),
            DoubleConv(in_channels, out_channels)
        )
    
    def forward(self, x):
        return self.maxpool_conv(x)


class Up(nn.Module):
    """Upsampling block for U-Net decoder."""
    def __init__(self, in_channels, out_channels, bilinear=True):
        super(Up, self).__init__()
        
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)
    
    def forward(self, x1, x2):
        x1 = self.up(x1)
        
        # Handle size mismatch
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, [diffX // 2, diffX - diffX // 2,
                        diffY // 2, diffY - diffY // 2])
        
        # Concatenate skip connection
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)


class UNet(nn.Module):
    """
    U-Net architecture for image reconstruction refinement.
    
    Args:
        in_channels: Number of input channels (default: 1)
        out_channels: Number of output channels (default: 1)
        base_features: Number of features in first layer (default: 64)
        bilinear: Use bilinear upsampling (default: True)
    """
    def __init__(self, in_channels=1, out_channels=1, base_features=64, bilinear=True):
        super(UNet, self).__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bilinear = bilinear
        
        factor = 2 if bilinear else 1
        
        # Encoder
        self.inc = DoubleConv(in_channels, base_features)
        self.down1 = Down(base_features, base_features * 2)
        self.down2 = Down(base_features * 2, base_features * 4)
        self.down3 = Down(base_features * 4, base_features * 8)
        self.down4 = Down(base_features * 8, base_features * 16 // factor)
        
        # Decoder
        self.up1 = Up(base_features * 16, base_features * 8 // factor, bilinear)
        self.up2 = Up(base_features * 8, base_features * 4 // factor, bilinear)
        self.up3 = Up(base_features * 4, base_features * 2 // factor, bilinear)
        self.up4 = Up(base_features * 2, base_features, bilinear)
        
        # Output
        self.outc = nn.Conv2d(base_features, out_channels, kernel_size=1)
    
    def forward(self, x):
        # Encoder
        x1 = self.inc(x)
        x2 = self.down1(x1)
        x3 = self.down2(x2)
        x4 = self.down3(x3)
        x5 = self.down4(x4)
        
        # Decoder with skip connections
        x = self.up1(x5, x4)
        x = self.up2(x, x3)
        x = self.up3(x, x2)
        x = self.up4(x, x1)
        
        # Output
        logits = self.outc(x)
        return logits


class Generator(nn.Module):
    """
    Generator (Decoder) for reconstructing images from measurements.
    
    Architecture:
    1. Standardization: Instance Normalization (no affine parameters)
    2. Initial Mapping: Linear + BatchNorm + Mish
    3. Reshape to image
    4. Refinement: U-Net
    
    Args:
        n_measurements: Number of measurements (m)
        n_pixels: Number of pixels in the image (n = H * W)
        img_size: Image size (H = W)
        base_features: Base features for U-Net (default: 64)
    """
    def __init__(self, n_measurements, n_pixels, img_size, base_features=64):
        super(Generator, self).__init__()
        
        self.n_measurements = n_measurements
        self.n_pixels = n_pixels
        self.img_size = img_size
        
        # Standardization: Instance Normalization without affine parameters
        self.standardization = nn.InstanceNorm1d(1, affine=False)
        
        # Initial Mapping: m -> n
        self.initial_mapping = nn.Sequential(
            nn.Linear(n_measurements, n_pixels),
            nn.BatchNorm1d(n_pixels),
            Mish()
        )
        
        # Refinement Network: U-Net
        self.unet = UNet(in_channels=1, out_channels=1, base_features=base_features)
    
    def forward(self, measurements):
        """
        Forward pass: reconstruct image from measurements.
        
        Args:
            measurements: Tensor of shape (B, m)
        
        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        batch_size = measurements.shape[0]
        
        # Standardization: (B, m) -> (B, 1, m) -> InstanceNorm -> (B, m)
        x = measurements.unsqueeze(1)
        x = self.standardization(x)
        x = x.squeeze(1)
        
        # Initial Mapping: (B, m) -> (B, n)
        x = self.initial_mapping(x)
        
        # Reshape to image: (B, n) -> (B, 1, H, W)
        x = x.view(batch_size, 1, self.img_size, self.img_size)
        
        # Refinement with U-Net
        x = self.unet(x)
        
        # Ensure output is in valid range [0, 1]
        x = torch.sigmoid(x)
        
        return x


class Discriminator(nn.Module):
    """
    PatchGAN Discriminator for GAN-based training.
    
    Uses a series of convolutions to classify patches as real or fake.
    Output is a map of predictions for overlapping patches.
    
    Args:
        in_channels: Number of input channels (default: 1)
        base_features: Number of features in first layer (default: 64)
        n_layers: Number of discriminator layers (default: 3)
    """
    def __init__(self, in_channels=1, base_features=64, n_layers=3):
        super(Discriminator, self).__init__()
        
        layers = []
        
        # First layer (no normalization)
        layers.append(nn.Conv2d(in_channels, base_features, kernel_size=4, stride=2, padding=1))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        nf_mult = 1
        nf_mult_prev = 1
        
        # Intermediate layers
        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            layers.append(nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                                   kernel_size=4, stride=2, padding=1, bias=False))
            layers.append(nn.BatchNorm2d(base_features * nf_mult))
            layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Second to last layer
        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        layers.append(nn.Conv2d(base_features * nf_mult_prev, base_features * nf_mult,
                               kernel_size=4, stride=1, padding=1, bias=False))
        layers.append(nn.BatchNorm2d(base_features * nf_mult))
        layers.append(nn.LeakyReLU(0.2, inplace=True))
        
        # Output layer (no sigmoid - use with BCEWithLogitsLoss or LSGAN loss)
        layers.append(nn.Conv2d(base_features * nf_mult, 1, kernel_size=4, stride=1, padding=1))
        
        self.model = nn.Sequential(*layers)
    
    def forward(self, x):
        """
        Forward pass.
        
        Args:
            x: Input image (B, 1, H, W)
        
        Returns:
            Discrimination map (B, 1, H', W')
        """
        return self.model(x)


class SPIModel(nn.Module):
    """
    SPI (Single-Pixel Imaging) model with fixed patterns.
    
    Uses pre-generated patterns (Hadamard zigzag or Random binary) for measurement,
    and a trainable Generator (Decoder) for reconstruction.
    
    Args:
        img_size: Image size (H = W, default: 128)
        n_measurements: Number of measurements (default: 1024)
        noise_std: Standard deviation of Gaussian noise (default: 0.05)
        base_features: Base features for U-Net decoder (default: 64)
        pattern_type: Type of patterns ('hadamard' or 'random')
        seed: Random seed for random patterns
    """
    def __init__(self, img_size=128, n_measurements=1024, noise_std=0.05, 
                 base_features=64, pattern_type='hadamard', seed=42):
        super(SPIModel, self).__init__()
        
        self.img_size = img_size
        self.n_pixels = img_size * img_size
        self.n_measurements = n_measurements
        self.noise_std = noise_std
        self.pattern_type = pattern_type
        
        # Fixed Pattern Encoder (not trainable)
        self.encoder = FixedPatternEncoder(
            img_size=img_size,
            n_measurements=n_measurements,
            pattern_type=pattern_type,
            seed=seed
        )
        
        # Generator (Decoder) - trainable
        self.generator = Generator(
            n_measurements=n_measurements,
            n_pixels=self.n_pixels,
            img_size=img_size,
            base_features=base_features
        )
    
    def forward(self, x, add_noise=True):
        """
        Forward pass through the SPI model.
        
        Args:
            x: Input image batch of shape (B, 1, H, W)
            add_noise: Whether to add Gaussian noise to measurements
        
        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        # Encode: get measurements using fixed patterns
        measurements = self.encoder(x)
        
        # Add Gaussian noise (during training)
        if add_noise and self.training and self.noise_std > 0:
            noise = torch.randn_like(measurements) * self.noise_std
            measurements = measurements + noise
        
        # Decode: reconstruct image
        reconstructed = self.generator(measurements)
        
        return reconstructed
    
    def encode(self, x, add_noise=False):
        """
        Encode image to measurements.
        
        Args:
            x: Input image batch of shape (B, 1, H, W)
            add_noise: Whether to add noise
        
        Returns:
            measurements: Tensor of shape (B, m)
        """
        measurements = self.encoder(x)
        if add_noise and self.noise_std > 0:
            noise = torch.randn_like(measurements) * self.noise_std
            measurements = measurements + noise
        return measurements
    
    def decode(self, measurements):
        """
        Decode measurements to image.
        
        Args:
            measurements: Tensor of shape (B, m)
        
        Returns:
            Reconstructed image of shape (B, 1, H, W)
        """
        return self.generator(measurements)
    
    def get_patterns(self):
        """Return the fixed pattern matrix."""
        return self.encoder.get_patterns()
    
    def get_compression_ratio(self):
        """Calculate compression ratio: n_pixels / n_measurements."""
        return self.n_pixels / self.n_measurements
    
    def get_sampling_ratio(self):
        """Calculate sampling ratio: n_measurements / n_pixels."""
        return self.n_measurements / self.n_pixels


def create_spi_model(img_size=128, n_measurements=1024, noise_std=0.05, 
                     base_features=64, pattern_type='hadamard', seed=42):
    """
    Create SPI model with specified configuration.
    
    Args:
        img_size: Image size (H = W)
        n_measurements: Number of measurements (default: 1024)
        noise_std: Standard deviation of Gaussian noise (default: 0.05)
        base_features: Base features for U-Net decoder
        pattern_type: 'hadamard' or 'random'
        seed: Random seed for random patterns
    
    Returns:
        SPIModel instance
    """
    n_pixels = img_size * img_size
    sampling_ratio = n_measurements / n_pixels
    
    model = SPIModel(
        img_size=img_size,
        n_measurements=n_measurements,
        noise_std=noise_std,
        base_features=base_features,
        pattern_type=pattern_type,
        seed=seed
    )
    
    print(f"Created SPI model:")
    print(f"  Image size: {img_size}x{img_size}")
    print(f"  Pixels: {n_pixels}")
    print(f"  Measurements: {n_measurements}")
    print(f"  Sampling ratio: {sampling_ratio*100:.2f}%")
    print(f"  Compression ratio: {1/sampling_ratio:.2f}x")
    print(f"  Noise std: {noise_std}")
    print(f"  Pattern type: {pattern_type}")
    
    return model


def create_discriminator(in_channels=1, base_features=64, n_layers=3):
    """
    Create PatchGAN discriminator.
    
    Args:
        in_channels: Number of input channels
        base_features: Base features
        n_layers: Number of layers
    
    Returns:
        Discriminator instance
    """
    return Discriminator(in_channels=in_channels, base_features=base_features, n_layers=n_layers)


# Backward compatibility aliases
Decoder = Generator


if __name__ == "__main__":
    # Test the model
    print("Testing SPI model components...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Test parameters
    img_size = 128
    batch_size = 4
    n_measurements = 1024
    noise_std = 0.05
    
    # Create test input
    x = torch.rand(batch_size, 1, img_size, img_size).to(device)
    print(f"\nInput shape: {x.shape}")
    
    # Test Hadamard patterns
    print("\n--- Testing Hadamard Zigzag Patterns ---")
    encoder_hadamard = FixedPatternEncoder(img_size, n_measurements, 'hadamard').to(device)
    measurements = encoder_hadamard(x)
    print(f"Measurements shape: {measurements.shape}")
    print(f"Pattern values: {torch.unique(encoder_hadamard.get_patterns())}")
    
    # Test Random patterns
    print("\n--- Testing Random Binary Patterns ---")
    encoder_random = FixedPatternEncoder(img_size, n_measurements, 'random').to(device)
    measurements_rand = encoder_random(x)
    print(f"Measurements shape: {measurements_rand.shape}")
    print(f"Pattern values: {torch.unique(encoder_random.get_patterns())}")
    
    # Test Generator
    print("\n--- Testing Generator ---")
    generator = Generator(n_measurements, img_size * img_size, img_size).to(device)
    reconstructed = generator(measurements)
    print(f"Reconstructed shape: {reconstructed.shape}")
    
    # Test Discriminator
    print("\n--- Testing Discriminator ---")
    discriminator = Discriminator(in_channels=1, base_features=64, n_layers=3).to(device)
    disc_out = discriminator(x)
    print(f"Discriminator output shape: {disc_out.shape}")
    
    # Test full SPI model
    print("\n--- Testing Full SPI Model ---")
    model = SPIModel(img_size=img_size, n_measurements=n_measurements, 
                     noise_std=noise_std, pattern_type='hadamard').to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    encoder_params = sum(p.numel() for p in model.encoder.parameters())
    generator_params = sum(p.numel() for p in model.generator.parameters())
    disc_params = sum(p.numel() for p in discriminator.parameters())
    
    print(f"Total model parameters: {total_params:,}")
    print(f"Encoder parameters (fixed, non-trainable): {encoder_params:,}")
    print(f"Generator parameters (trainable): {generator_params:,}")
    print(f"Discriminator parameters: {disc_params:,}")
    
    # Forward pass
    model.train()
    output = model(x, add_noise=True)
    print(f"Output shape: {output.shape}")
    print(f"Output range: [{output.min():.4f}, {output.max():.4f}]")
    
    # Test with sampling ratio
    print("\n--- Testing create_spi_model ---")
    model2 = create_spi_model(img_size=128, n_measurements=1024, pattern_type='random')
    print(f"Sampling ratio: {model2.get_sampling_ratio()*100:.2f}%")
    print(f"Compression ratio: {model2.get_compression_ratio():.2f}x")
    
    print("\nmodel.py: OK")
