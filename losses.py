"""
Loss functions for SPI (Single-Pixel Imaging) with GAN-based training

Implements:
1. L1 + SSIM combined reconstruction loss
2. Perceptual loss using VGG features
3. FFT frequency domain loss
4. GAN losses (LSGAN, Vanilla GAN)
5. Combined Generator and Discriminator losses
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models


class SSIM(nn.Module):
    """
    Structural Similarity Index (SSIM) loss module.
    
    Args:
        window_size: Size of the Gaussian window
        size_average: If True, returns the mean of SSIM; otherwise returns full map
        channel: Number of input channels
    """
    def __init__(self, window_size=11, size_average=True, channel=1):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = channel
        self.window = self._create_window(window_size, channel)
    
    def _gaussian(self, window_size, sigma):
        """Create 1D Gaussian kernel."""
        gauss = torch.Tensor([
            torch.exp(torch.tensor(-(x - window_size//2)**2 / (2*sigma**2)))
            for x in range(window_size)
        ])
        return gauss / gauss.sum()
    
    def _create_window(self, window_size, channel):
        """Create 2D Gaussian window for SSIM."""
        _1D_window = self._gaussian(window_size, 1.5).unsqueeze(1)
        _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
        window = _2D_window.expand(channel, 1, window_size, window_size).contiguous()
        return window
    
    def forward(self, img1, img2):
        """
        Compute SSIM between two images.
        
        Args:
            img1, img2: Input images of shape (B, C, H, W)
        
        Returns:
            SSIM value (higher is better, range [0, 1])
        """
        (_, channel, _, _) = img1.size()
        
        # Move window to same device as input
        if self.window.device != img1.device:
            self.window = self.window.to(img1.device)
        
        if channel != self.channel:
            self.window = self._create_window(self.window_size, channel).to(img1.device)
            self.channel = channel
        
        return self._ssim(img1, img2, self.window, self.window_size, channel, self.size_average)
    
    def _ssim(self, img1, img2, window, window_size, channel, size_average=True):
        """Internal SSIM computation."""
        mu1 = F.conv2d(img1, window, padding=window_size//2, groups=channel)
        mu2 = F.conv2d(img2, window, padding=window_size//2, groups=channel)
        
        mu1_sq = mu1.pow(2)
        mu2_sq = mu2.pow(2)
        mu1_mu2 = mu1 * mu2
        
        sigma1_sq = F.conv2d(img1*img1, window, padding=window_size//2, groups=channel) - mu1_sq
        sigma2_sq = F.conv2d(img2*img2, window, padding=window_size//2, groups=channel) - mu2_sq
        sigma12 = F.conv2d(img1*img2, window, padding=window_size//2, groups=channel) - mu1_mu2
        
        C1 = 0.01**2
        C2 = 0.03**2
        
        ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2)) / ((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))
        
        if size_average:
            return ssim_map.mean()
        else:
            return ssim_map.mean(1).mean(1).mean(1)


class VGGPerceptualLoss(nn.Module):
    """
    Perceptual loss using VGG19 features.
    
    Computes L1 distance between VGG feature representations.
    Uses features from relu1_2, relu2_2, relu3_4, relu4_4.
    
    Args:
        feature_layers: List of VGG layer indices to extract features from
        weights: Weights for each feature layer loss
    """
    def __init__(self, feature_layers=[3, 8, 17, 26], weights=[1.0, 1.0, 1.0, 1.0]):
        super(VGGPerceptualLoss, self).__init__()
        
        # Load pretrained VGG19
        vgg = models.vgg19(weights=models.VGG19_Weights.IMAGENET1K_V1)
        
        # Extract feature layers
        self.feature_layers = feature_layers
        self.weights = weights
        
        # Create feature extractor modules
        self.features = nn.ModuleList()
        prev_layer = 0
        for layer in feature_layers:
            self.features.append(
                nn.Sequential(*list(vgg.features.children())[prev_layer:layer+1])
            )
            prev_layer = layer + 1
        
        # Freeze VGG weights
        for param in self.parameters():
            param.requires_grad = False
        
        # ImageNet normalization
        self.register_buffer('mean', torch.tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1))
        self.register_buffer('std', torch.tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1))
    
    def _normalize(self, x):
        """Normalize input for VGG (convert grayscale to RGB and normalize)."""
        # Convert grayscale to RGB by repeating channels
        if x.shape[1] == 1:
            x = x.repeat(1, 3, 1, 1)
        # Move buffers to same device as input if needed
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        # Normalize with ImageNet stats
        return (x - self.mean) / self.std
    
    def forward(self, pred, target):
        """
        Compute perceptual loss.
        
        Args:
            pred: Predicted image (B, 1, H, W)
            target: Ground truth image (B, 1, H, W)
        
        Returns:
            Perceptual loss value
        """
        # Move VGG features to same device as input if needed
        if next(self.features.parameters()).device != pred.device:
            self.features = self.features.to(pred.device)
        
        pred = self._normalize(pred)
        target = self._normalize(target)
        
        loss = 0.0
        pred_feat = pred
        target_feat = target
        
        for i, feature_extractor in enumerate(self.features):
            pred_feat = feature_extractor(pred_feat)
            target_feat = feature_extractor(target_feat)
            loss += self.weights[i] * F.l1_loss(pred_feat, target_feat)
        
        return loss


class FFTLoss(nn.Module):
    """
    Frequency domain loss using FFT.
    
    Computes L1 distance between frequency representations.
    This helps preserve high-frequency details like edges and textures.
    
    Args:
        loss_type: Type of FFT loss ('amplitude', 'phase', 'both')
    """
    def __init__(self, loss_type='amplitude'):
        super(FFTLoss, self).__init__()
        self.loss_type = loss_type
    
    def forward(self, pred, target):
        """
        Compute FFT loss.
        
        Args:
            pred: Predicted image (B, 1, H, W)
            target: Ground truth image (B, 1, H, W)
        
        Returns:
            FFT loss value
        """
        # Compute 2D FFT
        pred_fft = torch.fft.fft2(pred, norm='ortho')
        target_fft = torch.fft.fft2(target, norm='ortho')
        
        if self.loss_type == 'amplitude':
            # Amplitude (magnitude) loss
            pred_amp = torch.abs(pred_fft)
            target_amp = torch.abs(target_fft)
            loss = F.l1_loss(pred_amp, target_amp)
        elif self.loss_type == 'phase':
            # Phase loss
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            loss = F.l1_loss(pred_phase, target_phase)
        else:  # 'both'
            pred_amp = torch.abs(pred_fft)
            target_amp = torch.abs(target_fft)
            pred_phase = torch.angle(pred_fft)
            target_phase = torch.angle(target_fft)
            loss = F.l1_loss(pred_amp, target_amp) + 0.1 * F.l1_loss(pred_phase, target_phase)
        
        return loss


class CharbonnierLoss(nn.Module):
    """
    Charbonnier loss (smooth L1 loss).
    
    L(x, y) = sqrt((x - y)^2 + eps^2)
    
    More robust to outliers than L2, smoother than L1.
    """
    def __init__(self, eps=1e-6):
        super(CharbonnierLoss, self).__init__()
        self.eps = eps
    
    def forward(self, pred, target):
        diff = pred - target
        loss = torch.sqrt(diff**2 + self.eps**2)
        return loss.mean()


# ===================== GAN Losses =====================

class GANLoss(nn.Module):
    """
    GAN loss supporting multiple GAN types.
    
    Supports:
    - 'vanilla': Original GAN with BCE loss
    - 'lsgan': Least Squares GAN (more stable)
    - 'wgan': Wasserstein GAN
    
    Args:
        gan_mode: Type of GAN loss ('vanilla', 'lsgan', 'wgan')
        target_real_label: Label for real images (default: 1.0)
        target_fake_label: Label for fake images (default: 0.0)
    """
    def __init__(self, gan_mode='lsgan', target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        
        self.gan_mode = gan_mode
        
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode == 'wgan':
            self.loss = None  # Wasserstein loss is computed directly
        else:
            raise NotImplementedError(f'GAN mode {gan_mode} not implemented')
    
    def get_target_tensor(self, prediction, target_is_real):
        """Create target tensor with same shape as prediction."""
        if target_is_real:
            target_tensor = self.real_label.to(prediction.device)
        else:
            target_tensor = self.fake_label.to(prediction.device)
        return target_tensor.expand_as(prediction)
    
    def forward(self, prediction, target_is_real):
        """
        Compute GAN loss.
        
        Args:
            prediction: Discriminator output
            target_is_real: Whether target is real (True) or fake (False)
        
        Returns:
            GAN loss value
        """
        if self.gan_mode == 'wgan':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        else:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        
        return loss


class ReconstructionLoss(nn.Module):
    """
    Combined reconstruction loss for generator.
    
    Loss = w1 * L1 + w2 * (1-SSIM) + w3 * Perceptual + w4 * FFT
    
    Args:
        w1: Weight for L1/Charbonnier loss (default: 0.15)
        w2: Weight for SSIM loss (default: 0.5)
        w3: Weight for Perceptual loss (default: 0.25)
        w4: Weight for FFT loss (default: 0.1)
        use_charbonnier: Use Charbonnier instead of L1 (default: True)
        use_perceptual: Enable perceptual loss (default: True)
        use_fft: Enable FFT loss (default: True)
    """
    def __init__(self, w1=0.15, w2=0.5, w3=0.25, w4=0.1,
                 use_charbonnier=True, use_perceptual=True, use_fft=True):
        super(ReconstructionLoss, self).__init__()
        
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        
        self.use_charbonnier = use_charbonnier
        self.use_perceptual = use_perceptual
        self.use_fft = use_fft
        
        # Pixel loss
        if use_charbonnier:
            self.pixel_loss = CharbonnierLoss()
        else:
            self.pixel_loss = nn.L1Loss()
        
        # SSIM loss
        self.ssim = SSIM(window_size=11, channel=1)
        
        # Perceptual loss
        self.perceptual_loss = None
        if use_perceptual:
            self.perceptual_loss = VGGPerceptualLoss()
        
        # FFT loss
        self.fft_loss = None
        if use_fft:
            self.fft_loss = FFTLoss(loss_type='amplitude')
    
    def forward(self, pred, target):
        """
        Compute reconstruction loss.
        
        Args:
            pred: Predicted image (B, 1, H, W)
            target: Ground truth image (B, 1, H, W)
        
        Returns:
            total_loss, loss_dict with individual components
        """
        losses = {}
        
        # Pixel loss
        pixel_loss = self.pixel_loss(pred, target)
        losses['pixel'] = pixel_loss
        
        # SSIM loss
        ssim_val = self.ssim(pred, target)
        ssim_loss = 1 - ssim_val
        losses['ssim'] = ssim_val
        losses['ssim_loss'] = ssim_loss
        
        # Total reconstruction loss
        total = self.w1 * pixel_loss + self.w2 * ssim_loss
        
        # Perceptual loss
        if self.use_perceptual and self.perceptual_loss is not None:
            perc_loss = self.perceptual_loss(pred, target)
            losses['perceptual'] = perc_loss
            total += self.w3 * perc_loss
        
        # FFT loss
        if self.use_fft and self.fft_loss is not None:
            fft_loss = self.fft_loss(pred, target)
            losses['fft'] = fft_loss
            total += self.w4 * fft_loss
        
        losses['total_recon'] = total
        
        return total, losses


class GeneratorLoss(nn.Module):
    """
    Complete Generator loss for GAN training.
    
    Loss = w_recon * ReconstructionLoss + w_adv * AdversarialLoss
    
    Args:
        w_recon: Weight for reconstruction loss (default: 100.0)
        w_adv: Weight for adversarial loss (default: 1.0)
        gan_mode: GAN mode ('lsgan', 'vanilla', 'wgan')
        recon_config: Dict with reconstruction loss config
    """
    def __init__(self, w_recon=100.0, w_adv=1.0, gan_mode='lsgan',
                 w1=0.15, w2=0.5, w3=0.25, w4=0.1,
                 use_perceptual=True, use_fft=True):
        super(GeneratorLoss, self).__init__()
        
        self.w_recon = w_recon
        self.w_adv = w_adv
        
        # Reconstruction loss
        self.recon_loss = ReconstructionLoss(
            w1=w1, w2=w2, w3=w3, w4=w4,
            use_perceptual=use_perceptual, use_fft=use_fft
        )
        
        # Adversarial loss
        self.gan_loss = GANLoss(gan_mode=gan_mode)
    
    def forward(self, fake_img, real_img, disc_fake):
        """
        Compute total generator loss.
        
        Args:
            fake_img: Generated image
            real_img: Real (target) image
            disc_fake: Discriminator output for fake image
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        
        # Reconstruction loss
        recon_loss, recon_dict = self.recon_loss(fake_img, real_img)
        losses.update(recon_dict)
        
        # Adversarial loss (generator wants discriminator to think fake is real)
        adv_loss = self.gan_loss(disc_fake, target_is_real=True)
        losses['adv_g'] = adv_loss
        
        # Total generator loss
        total = self.w_recon * recon_loss + self.w_adv * adv_loss
        losses['total_g'] = total
        
        return total, losses


class DiscriminatorLoss(nn.Module):
    """
    Discriminator loss for GAN training.
    
    Loss = 0.5 * (Loss_real + Loss_fake)
    
    Args:
        gan_mode: GAN mode ('lsgan', 'vanilla', 'wgan')
    """
    def __init__(self, gan_mode='lsgan'):
        super(DiscriminatorLoss, self).__init__()
        self.gan_loss = GANLoss(gan_mode=gan_mode)
    
    def forward(self, disc_real, disc_fake):
        """
        Compute discriminator loss.
        
        Args:
            disc_real: Discriminator output for real images
            disc_fake: Discriminator output for fake images
        
        Returns:
            total_loss, loss_dict
        """
        losses = {}
        
        # Loss on real images (discriminator wants to classify as real)
        loss_real = self.gan_loss(disc_real, target_is_real=True)
        losses['d_real'] = loss_real
        
        # Loss on fake images (discriminator wants to classify as fake)
        loss_fake = self.gan_loss(disc_fake, target_is_real=False)
        losses['d_fake'] = loss_fake
        
        # Total discriminator loss
        total = 0.5 * (loss_real + loss_fake)
        losses['total_d'] = total
        
        return total, losses


def compute_psnr(pred, target, data_range=1.0):
    """
    Compute PSNR between predicted and target images.
    
    Args:
        pred: Predicted image
        target: Target image
        data_range: Data range (1.0 for [0,1] images)
    
    Returns:
        PSNR value in dB
    """
    mse = F.mse_loss(pred, target)
    if mse == 0:
        return float('inf')
    psnr = 10 * torch.log10(data_range ** 2 / mse)
    return psnr.item()


if __name__ == "__main__":
    # Test loss functions
    print("Testing loss functions...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Test data
    pred = torch.rand(4, 1, 128, 128).to(device)
    target = torch.rand(4, 1, 128, 128).to(device)
    
    # Test SSIM
    print("\n--- Testing SSIM ---")
    ssim = SSIM()
    ssim_val = ssim(pred, target)
    print(f"SSIM value: {ssim_val.item():.4f}")
    
    # Test ReconstructionLoss
    print("\n--- Testing ReconstructionLoss ---")
    recon_loss = ReconstructionLoss(use_perceptual=False, use_fft=True).to(device)
    loss, loss_dict = recon_loss(pred, target)
    print(f"Reconstruction loss: {loss.item():.4f}")
    for k, v in loss_dict.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k}: {val:.4f}")
    
    # Test GANLoss
    print("\n--- Testing GANLoss ---")
    gan_loss = GANLoss(gan_mode='lsgan').to(device)
    disc_output = torch.rand(4, 1, 14, 14).to(device)
    loss_real = gan_loss(disc_output, target_is_real=True)
    loss_fake = gan_loss(disc_output, target_is_real=False)
    print(f"GAN loss (real): {loss_real.item():.4f}")
    print(f"GAN loss (fake): {loss_fake.item():.4f}")
    
    # Test GeneratorLoss
    print("\n--- Testing GeneratorLoss ---")
    g_loss = GeneratorLoss(w_recon=100.0, w_adv=1.0, use_perceptual=False).to(device)
    total_g, g_dict = g_loss(pred, target, disc_output)
    print(f"Total generator loss: {total_g.item():.4f}")
    for k, v in g_dict.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k}: {val:.4f}")
    
    # Test DiscriminatorLoss
    print("\n--- Testing DiscriminatorLoss ---")
    d_loss = DiscriminatorLoss(gan_mode='lsgan').to(device)
    disc_real = torch.rand(4, 1, 14, 14).to(device)
    disc_fake = torch.rand(4, 1, 14, 14).to(device)
    total_d, d_dict = d_loss(disc_real, disc_fake)
    print(f"Total discriminator loss: {total_d.item():.4f}")
    for k, v in d_dict.items():
        val = v.item() if torch.is_tensor(v) else v
        print(f"  {k}: {val:.4f}")
    
    # Test PSNR
    print("\n--- Testing PSNR ---")
    psnr = compute_psnr(pred, target)
    print(f"PSNR: {psnr:.2f} dB")
    
    print("\nlosses.py: OK")
