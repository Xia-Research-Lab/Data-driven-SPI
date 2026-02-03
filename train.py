"""
Training script for SPI (Single-Pixel Imaging) with GAN-based reconstruction

Features:
1. Fixed measurement patterns (Hadamard zigzag or Random binary)
2. Generator (Decoder) training with reconstruction + adversarial loss
3. PatchGAN Discriminator training
4. Alternating training of Generator and Discriminator
5. Noise injection with std=0.05
6. BF16 mixed precision training for efficiency
"""

import os
import time
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.tensorboard import SummaryWriter
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

from model import SPIModel, Discriminator, create_spi_model, create_discriminator
from losses import GeneratorLoss, DiscriminatorLoss, compute_psnr, SSIM
from utils import (
    get_dataloaders, save_checkpoint, 
    load_checkpoint, EarlyStopping, count_parameters
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description='Train SPI model with GAN')
    
    # Data
    parser.add_argument('--data_dir', type=str, default='cyto128',
                        help='Path to dataset directory')
    parser.add_argument('--img_size', type=int, default=128,
                        help='Image size (H=W)')
    
    # Model
    parser.add_argument('--n_measurements', type=int, default=2048,
                        help='Number of measurements (fixed at 1024)')
    parser.add_argument('--noise_std', type=float, default=0.05,
                        help='Noise standard deviation')
    parser.add_argument('--base_features', type=int, default=64,
                        help='Base features for U-Net')
    parser.add_argument('--pattern_type', type=str, default='hadamard',
                        choices=['hadamard', 'random'],
                        help='Pattern type: hadamard or random')
    
    # Discriminator
    parser.add_argument('--d_base_features', type=int, default=64,
                        help='Base features for discriminator')
    parser.add_argument('--d_n_layers', type=int, default=3,
                        help='Number of discriminator layers')
    
    # Training
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size')
    parser.add_argument('--lr_g', type=float, default=2e-4,
                        help='Generator learning rate')
    parser.add_argument('--lr_d', type=float, default=2e-4,
                        help='Discriminator learning rate')
    parser.add_argument('--beta1', type=float, default=0.5,
                        help='Adam beta1')
    parser.add_argument('--beta2', type=float, default=0.999,
                        help='Adam beta2')
    
    # Loss weights
    parser.add_argument('--w_recon', type=float, default=100.0,
                        help='Weight for reconstruction loss')
    parser.add_argument('--w_adv', type=float, default=1.0,
                        help='Weight for adversarial loss')
    parser.add_argument('--w1', type=float, default=0.15,
                        help='Weight for pixel loss (L1/Charbonnier)')
    parser.add_argument('--w2', type=float, default=0.0,
                        help='Weight for SSIM loss')
    parser.add_argument('--w3', type=float, default=0.0,
                        help='Weight for perceptual loss')
    parser.add_argument('--w4', type=float, default=0.0,
                        help='Weight for FFT loss')
    
    # GAN config
    parser.add_argument('--gan_mode', type=str, default='lsgan',
                        choices=['lsgan', 'vanilla', 'wgan'],
                        help='GAN loss mode')
    parser.add_argument('--d_update_freq', type=int, default=1,
                        help='Update discriminator every N generator updates')
    
    # Loss configuration
    parser.add_argument('--use_perceptual', action='store_true', default=False,
                        help='Use perceptual loss')
    parser.add_argument('--use_fft', action='store_true', default=False,
                        help='Use FFT frequency loss')
    parser.add_argument('--no_perceptual', action='store_true',
                        help='Disable perceptual loss')
    parser.add_argument('--no_fft', action='store_true',
                        help='Disable FFT loss')
    parser.add_argument('--no_gan', action='store_true',
                        help='Disable GAN training (reconstruction only)')
    
    # Misc
    parser.add_argument('--num_workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--save_dir', type=str, default='checkpoints',
                        help='Directory to save checkpoints')
    parser.add_argument('--log_dir', type=str, default='logs',
                        help='Directory for tensorboard logs')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    parser.add_argument('--seed', type=int, default=42,
                        help='Random seed')
    
    return parser.parse_args()


def set_seed(seed):
    """Set random seed for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def train_one_epoch_gan(generator, discriminator, dataloader, 
                        g_criterion, d_criterion, g_optimizer, d_optimizer,
                        device, epoch, total_epochs, d_update_freq=1, use_gan=True,
                        scaler=None, use_bf16=True):
    """
    Train for one epoch with GAN and optional BF16 mixed precision.
    
    Returns:
        Dictionary with training metrics
    """
    generator.train()
    if use_gan:
        discriminator.train()
    
    metrics = {
        'g_loss': 0, 'recon_loss': 0, 'adv_g_loss': 0,
        'd_loss': 0, 'd_real': 0, 'd_fake': 0,
        'psnr': 0, 'ssim': 0
    }
    n_batches = 0
    
    ssim_fn = SSIM()
    
    pbar = tqdm(dataloader, desc=f'Epoch {epoch+1}/{total_epochs}')
    
    for batch_idx, images in enumerate(pbar):
        images = images.to(device)
        batch_size = images.shape[0]
        
        # ==================
        # Train Generator
        # ==================
        g_optimizer.zero_grad()
        
        # Forward pass with BF16 autocast
        with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
            # Generate fake images
            fake_images = generator(images, add_noise=True)
            
            if use_gan:
                # Get discriminator output for fake
                disc_fake = discriminator(fake_images)
                
                # Compute generator loss
                g_loss, g_loss_dict = g_criterion(fake_images, images, disc_fake)
            else:
                # No GAN - just reconstruction loss
                from losses import ReconstructionLoss
                recon_fn = g_criterion.recon_loss
                g_loss, g_loss_dict = recon_fn(fake_images, images)
                g_loss_dict['adv_g'] = torch.tensor(0.0)
                g_loss_dict['total_g'] = g_loss
        
        # Backward with scaler
        if scaler is not None:
            scaler.scale(g_loss).backward()
            scaler.step(g_optimizer)
        else:
            g_loss.backward()
            g_optimizer.step()
        
        # ==================
        # Train Discriminator
        # ==================
        d_loss_val = 0
        d_real_val = 0
        d_fake_val = 0
        
        if use_gan and (batch_idx + 1) % d_update_freq == 0:
            d_optimizer.zero_grad()
            
            with autocast(device_type='cuda', dtype=torch.bfloat16, enabled=use_bf16):
                # Real images
                disc_real = discriminator(images)
                
                # Fake images (detached - no gradient to generator)
                fake_images_detached = fake_images.detach()
                disc_fake_detached = discriminator(fake_images_detached)
                
                # Compute discriminator loss
                d_loss, d_loss_dict = d_criterion(disc_real, disc_fake_detached)
            
            # Backward with scaler
            if scaler is not None:
                scaler.scale(d_loss).backward()
                scaler.step(d_optimizer)
            else:
                d_loss.backward()
                d_optimizer.step()
            
            d_loss_val = d_loss.item()
            d_real_val = d_loss_dict['d_real'].item()
            d_fake_val = d_loss_dict['d_fake'].item()
        
        # Update scaler once per iteration
        if scaler is not None:
            scaler.update()
        
        # Compute metrics (convert to float32 for SSIM compatibility)
        with torch.no_grad():
            fake_images_fp32 = fake_images.float()
            images_fp32 = images.float()
            psnr = compute_psnr(fake_images_fp32, images_fp32)
            ssim_val = ssim_fn(fake_images_fp32, images_fp32).item()
        
        # Accumulate metrics
        metrics['g_loss'] += g_loss.item()
        metrics['recon_loss'] += g_loss_dict.get('total_recon', g_loss).item() if torch.is_tensor(g_loss_dict.get('total_recon', g_loss)) else g_loss_dict.get('total_recon', g_loss.item())
        metrics['adv_g_loss'] += g_loss_dict['adv_g'].item() if torch.is_tensor(g_loss_dict['adv_g']) else g_loss_dict['adv_g']
        metrics['d_loss'] += d_loss_val
        metrics['d_real'] += d_real_val
        metrics['d_fake'] += d_fake_val
        metrics['psnr'] += psnr
        metrics['ssim'] += ssim_val
        n_batches += 1
        
        # Update progress bar
        pbar.set_postfix({
            'G': f'{g_loss.item():.4f}',
            'D': f'{d_loss_val:.4f}',
            'psnr': f'{psnr:.2f}',
            'ssim': f'{ssim_val:.4f}'
        })
    
    # Average metrics
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def validate(generator, dataloader, device):
    """
    Validate the generator.
    
    Returns:
        Dictionary with validation metrics
    """
    generator.eval()
    
    metrics = {'psnr': 0, 'ssim': 0, 'l1': 0}
    n_batches = 0
    
    ssim_fn = SSIM()
    l1_fn = nn.L1Loss()
    
    with torch.no_grad():
        for images in tqdm(dataloader, desc='Validating'):
            images = images.to(device)
            
            # Generate without noise for validation
            fake_images = generator(images, add_noise=False)
            
            # Compute metrics
            metrics['psnr'] += compute_psnr(fake_images, images)
            metrics['ssim'] += ssim_fn(fake_images, images).item()
            metrics['l1'] += l1_fn(fake_images, images).item()
            n_batches += 1
    
    for key in metrics:
        metrics[key] /= n_batches
    
    return metrics


def main():
    args = parse_args()
    
    # Set seed
    set_seed(args.seed)
    
    # Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Handle flag overrides
    use_perceptual = args.use_perceptual and not args.no_perceptual
    use_fft = args.use_fft and not args.no_fft
    use_gan = not args.no_gan
    
    # Create directories
    os.makedirs(args.save_dir, exist_ok=True)
    os.makedirs(args.log_dir, exist_ok=True)
    
    # Data loaders
    dataloaders = get_dataloaders(
        data_dir=args.data_dir,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        img_size=args.img_size
    )
    train_loader = dataloaders['train']
    valid_loader = dataloaders['valid']
    test_loader = dataloaders['test']
    
    print(f"Train batches: {len(train_loader)}")
    print(f"Valid batches: {len(valid_loader)}")
    print(f"Test batches: {len(test_loader)}")
    
    # Create models
    generator = create_spi_model(
        img_size=args.img_size,
        n_measurements=args.n_measurements,
        noise_std=args.noise_std,
        base_features=args.base_features,
        pattern_type=args.pattern_type
    ).to(device)
    
    discriminator = None
    if use_gan:
        discriminator = create_discriminator(
            in_channels=1,
            base_features=args.d_base_features,
            n_layers=args.d_n_layers
        ).to(device)
    
    # Count parameters
    g_params = count_parameters(generator)
    print(f"Generator parameters: {g_params:,}")
    if discriminator:
        d_params = count_parameters(discriminator)
        print(f"Discriminator parameters: {d_params:,}")
    
    # Loss functions
    g_criterion = GeneratorLoss(
        w_recon=args.w_recon,
        w_adv=args.w_adv if use_gan else 0,
        gan_mode=args.gan_mode,
        w1=args.w1, w2=args.w2, w3=args.w3, w4=args.w4,
        use_perceptual=use_perceptual,
        use_fft=use_fft
    ).to(device)
    
    d_criterion = None
    if use_gan:
        d_criterion = DiscriminatorLoss(gan_mode=args.gan_mode).to(device)
    
    # Optimizers
    g_optimizer = optim.Adam(
        generator.generator.parameters(),  # Only train generator (decoder) part
        lr=args.lr_g,
        betas=(args.beta1, args.beta2)
    )
    
    d_optimizer = None
    if use_gan:
        d_optimizer = optim.Adam(
            discriminator.parameters(),
            lr=args.lr_d,
            betas=(args.beta1, args.beta2)
        )
    
    # Learning rate schedulers
    g_scheduler = optim.lr_scheduler.CosineAnnealingLR(g_optimizer, T_max=args.epochs)
    d_scheduler = None
    if use_gan:
        d_scheduler = optim.lr_scheduler.CosineAnnealingLR(d_optimizer, T_max=args.epochs)
    
    # Tensorboard writer
    writer = SummaryWriter(log_dir=args.log_dir)
    
    # Early stopping
    early_stopping = EarlyStopping(patience=15, mode='max')
    
    # BF16 mixed precision - use GradScaler for stability
    use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
    scaler = GradScaler() if use_bf16 else None
    if use_bf16:
        print("Using BF16 mixed precision training")
    
    # Resume training
    start_epoch = 0
    best_psnr = 0
    
    if args.resume:
        checkpoint = torch.load(args.resume, map_location=device)
        generator.load_state_dict(checkpoint['generator_state_dict'])
        if use_gan and 'discriminator_state_dict' in checkpoint:
            discriminator.load_state_dict(checkpoint['discriminator_state_dict'])
        g_optimizer.load_state_dict(checkpoint['g_optimizer_state_dict'])
        if use_gan and 'd_optimizer_state_dict' in checkpoint:
            d_optimizer.load_state_dict(checkpoint['d_optimizer_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_psnr = checkpoint.get('best_psnr', 0)
        print(f"Resumed from epoch {start_epoch}")
    
    # Training loop
    print("\n" + "="*60)
    print("Starting GAN training" if use_gan else "Starting reconstruction-only training")
    print("="*60 + "\n")
    
    for epoch in range(start_epoch, args.epochs):
        epoch_start = time.time()
        
        # Train
        train_metrics = train_one_epoch_gan(
            generator=generator,
            discriminator=discriminator,
            dataloader=train_loader,
            g_criterion=g_criterion,
            d_criterion=d_criterion,
            g_optimizer=g_optimizer,
            d_optimizer=d_optimizer,
            device=device,
            epoch=epoch,
            total_epochs=args.epochs,
            d_update_freq=args.d_update_freq,
            use_gan=use_gan,
            scaler=scaler,
            use_bf16=use_bf16
        )
        
        # Validate
        val_metrics = validate(generator, valid_loader, device)
        
        # Update schedulers
        g_scheduler.step()
        if d_scheduler:
            d_scheduler.step()
        
        epoch_time = time.time() - epoch_start
        
        # Print metrics
        print(f"\nEpoch {epoch+1}/{args.epochs} ({epoch_time:.1f}s)")
        print(f"  Train - G: {train_metrics['g_loss']:.4f}, D: {train_metrics['d_loss']:.4f}, "
              f"PSNR: {train_metrics['psnr']:.2f}, SSIM: {train_metrics['ssim']:.4f}")
        print(f"  Valid - PSNR: {val_metrics['psnr']:.2f}, SSIM: {val_metrics['ssim']:.4f}, "
              f"L1: {val_metrics['l1']:.4f}")
        
        # Log to tensorboard
        writer.add_scalar('Train/G_Loss', train_metrics['g_loss'], epoch)
        writer.add_scalar('Train/D_Loss', train_metrics['d_loss'], epoch)
        writer.add_scalar('Train/Recon_Loss', train_metrics['recon_loss'], epoch)
        writer.add_scalar('Train/PSNR', train_metrics['psnr'], epoch)
        writer.add_scalar('Train/SSIM', train_metrics['ssim'], epoch)
        writer.add_scalar('Valid/PSNR', val_metrics['psnr'], epoch)
        writer.add_scalar('Valid/SSIM', val_metrics['ssim'], epoch)
        writer.add_scalar('Valid/L1', val_metrics['l1'], epoch)
        writer.add_scalar('LR/Generator', g_optimizer.param_groups[0]['lr'], epoch)
        
        # Save checkpoint
        is_best = val_metrics['psnr'] > best_psnr
        if is_best:
            best_psnr = val_metrics['psnr']
        
        checkpoint = {
            'epoch': epoch,
            'generator_state_dict': generator.state_dict(),
            'g_optimizer_state_dict': g_optimizer.state_dict(),
            'best_psnr': best_psnr,
            'args': vars(args)
        }
        if use_gan:
            checkpoint['discriminator_state_dict'] = discriminator.state_dict()
            checkpoint['d_optimizer_state_dict'] = d_optimizer.state_dict()
        
        # Save latest
        torch.save(checkpoint, os.path.join(args.save_dir, 'latest_model.pth'))
        
        # Save best (generator only for smaller file size)
        if is_best:
            best_checkpoint = {
                'epoch': epoch,
                'generator_state_dict': generator.generator.state_dict(),  # Only generator (decoder) weights
                'best_psnr': best_psnr,
                'args': vars(args)
            }
            torch.save(best_checkpoint, os.path.join(args.save_dir, 'best_model.pth'))
            print(f"  New best model! PSNR: {best_psnr:.2f}")
        
        # Early stopping
        if early_stopping(val_metrics['psnr']):
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
    
    writer.close()
    
    # Final test
    print("\n" + "="*60)
    print("Testing on test set...")
    print("="*60)
    
    # Load best model (generator-only checkpoint)
    best_checkpoint = torch.load(os.path.join(args.save_dir, 'best_model.pth'), map_location=device)
    generator.generator.load_state_dict(best_checkpoint['generator_state_dict'])
    
    test_metrics = validate(generator, test_loader, device)
    print(f"\nTest Results:")
    print(f"  PSNR: {test_metrics['psnr']:.2f} dB")
    print(f"  SSIM: {test_metrics['ssim']:.4f}")
    print(f"  L1: {test_metrics['l1']:.4f}")
    
    # Save patterns
    patterns = generator.get_patterns().cpu().numpy()
    np.save(os.path.join(args.save_dir, 'patterns.npy'), patterns)
    print(f"\nPatterns saved to {os.path.join(args.save_dir, 'patterns.npy')}")
    
    print("\nTraining complete!")


if __name__ == "__main__":
    main()
