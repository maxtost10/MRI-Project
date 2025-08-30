# %%
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from torch.utils.data import Dataset, DataLoader, random_split
import wandb
import numpy as np
from numpy.fft import ifft2, fftshift, ifftshift
import h5py
import matplotlib.pyplot as plt
from typing import Tuple, Optional
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping, ModelSummary

class MRIDataset(Dataset):
    """PyTorch dataset for MRI reconstruction."""
    
    def __init__(self, h5_path: str, mode: str = 'train'):
        self.h5_path = h5_path
        self.mode = mode
        
        # Load dataset info
        with h5py.File(h5_path, 'r') as f:
            self.length = f[mode]['phantoms'].shape[0]
    
    def __len__(self):
        return self.length
    
    def __getitem__(self, idx):
        with h5py.File(self.h5_path, 'r') as f:
            group = f[self.mode]
            
            # Load data
            kspace_full = group['kspace_full'][idx]
            kspace_undersampled = group['kspace_undersampled'][idx]
            mask = group['masks'][idx]
            
            # Convert to torch tensors
            # Stack real and imaginary parts as channels
            kspace_full_tensor = torch.stack([
                torch.from_numpy(np.real(kspace_full)).float(),
                torch.from_numpy(np.imag(kspace_full)).float()
            ])
            
            kspace_us_tensor = torch.stack([
                torch.from_numpy(np.real(kspace_undersampled)).float(),
                torch.from_numpy(np.imag(kspace_undersampled)).float()
            ])
            
            mask_tensor = torch.from_numpy(mask.astype(np.float32))
            
        return kspace_us_tensor, kspace_full_tensor, mask_tensor

class AdaptiveUNet(nn.Module):
    """U-Net with acceleration-factor-aware architecture and built-in data consistency."""
    
    def __init__(self, in_channels: int = 2, out_channels: int = 2, acceleration: int = 4):
        super(AdaptiveUNet, self).__init__()
        
        self.acceleration = acceleration
        
        # Adaptive kernel sizes based on acceleration factor
        # For the main convolutions, use larger kernels to capture wider spatial relationships
        main_kernel = min(7, 2 * acceleration + 1)  # e.g., 7 for R=4, 9 for R=8
        
        # Encoder with adaptive kernels
        self.enc1 = self.conv_block(in_channels, 64, kernel_size=main_kernel)
        self.enc2 = self.conv_block(64, 128, kernel_size=main_kernel)
        self.enc3 = self.conv_block(128, 256, kernel_size=5)  # Slightly smaller for deeper layers
        self.enc4 = self.conv_block(256, 512, kernel_size=3)  # Standard for high-level features
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024, kernel_size=3)
        
        # Decoder with adaptive upsampling
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512, kernel_size=3)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256, kernel_size=5)
        
        # These layers should handle the acceleration pattern specifically
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128, kernel_size=main_kernel)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64, kernel_size=main_kernel)
        
        # Final layer with acceleration-specific kernel
        self.final = nn.Conv2d(64, out_channels, kernel_size=main_kernel, padding=main_kernel//2)
        
        # Pooling
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels: int, out_channels: int, kernel_size: int = 3):
        """Convolutional block with adaptive kernel size."""
        padding = kernel_size // 2
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x, mask):
        # Store input for data consistency
        input_kspace = x
        
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(self.pool(enc1))
        enc3 = self.enc3(self.pool(enc2))
        enc4 = self.enc4(self.pool(enc3))
        
        # Bottleneck
        bottleneck = self.bottleneck(self.pool(enc4))
        
        # Decoder
        dec4 = self.upconv4(bottleneck)
        dec4 = torch.cat([dec4, enc4], dim=1)
        dec4 = self.dec4(dec4)
        
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat([dec3, enc3], dim=1)
        dec3 = self.dec3(dec3)
        
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat([dec2, enc2], dim=1)
        dec2 = self.dec2(dec2)
        
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat([dec1, enc1], dim=1)
        dec1 = self.dec1(dec1)
        
        # Get prediction
        pred_kspace = self.final(dec1)
        
        mask_expanded = mask.unsqueeze(1).expand_as(pred_kspace)
        # Replace predicted values with measured values where available
        pred_kspace = pred_kspace * (1 - mask_expanded) + input_kspace * mask_expanded
        
        return pred_kspace


class MRIReconstructionLightning(pl.LightningModule):
    """PyTorch Lightning module for MRI reconstruction."""
    
    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 2,
        acceleration: int = 4,
        learning_rate: float = 1e-3,
        weight_decay: float = 1e-5,
    ):
        super().__init__()
        self.save_hyperparameters()
        
        self.model = AdaptiveUNet(in_channels, out_channels, acceleration)
        
        # Loss
        self.criterion = nn.MSELoss()
        
    def forward(self, x, mask):
        return self.model(x, mask)
    
    def _shared_step(self, batch, stage: str):
        kspace_us, kspace_full, mask = batch
        
        # Forward pass with data consistency built-in (mask always passed)
        pred_kspace = self.model(kspace_us, mask)
        
        # Compute loss
        loss = self.criterion(pred_kspace, kspace_full)
        
        # Log metrics
        self.log(f'{stage}_loss', loss, on_step=True, on_epoch=True, prog_bar=True)
        
        # Log additional metrics during validation
        if stage == 'val':
            # Compute PSNR
            mse = torch.mean((pred_kspace - kspace_full) ** 2)
            self.log('val_mse', mse, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss
    
    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, 'train')
    
    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, 'val')
    
    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, 'test')
    
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay
        )
        
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode='min',
            factor=0.5,
            patience=5,
        )
        
        return {
            'optimizer': optimizer,
            'lr_scheduler': {
                'scheduler': scheduler,
                'monitor': 'val_loss',
                'interval': 'epoch',
                'frequency': 1
            }
        }
    
    def on_train_epoch_end(self):
        # Log learning rate
        current_lr = self.optimizers().param_groups[0]['lr']
        self.log('learning_rate', current_lr, on_epoch=True)

class MRIDataModule(pl.LightningDataModule):
    """PyTorch Lightning data module for MRI reconstruction."""
    
    def __init__(
        self,
        dataset_path: str,
        batch_size: int = 8,
        num_workers: int = 4,
        pin_memory: bool = True,
        val_split: float = 0.2
    ):
        super().__init__()
        self.dataset_path = dataset_path
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.val_split = val_split
    
    def setup(self, stage: Optional[str] = None):
        if stage == 'fit' or stage is None:
            full_train_dataset = MRIDataset(self.dataset_path, mode='train')
            train_size = int((1 - self.val_split) * len(full_train_dataset))
            val_size = len(full_train_dataset) - train_size
            
            self.train_dataset, self.val_dataset = random_split(
                full_train_dataset, 
                [train_size, val_size],
                generator=torch.Generator().manual_seed(42)  # For reproducible splits
            )
            
            print(f"Dataset split: {train_size} train, {val_size} validation")
        
        if stage == 'test' or stage is None:
            self.test_dataset = MRIDataset(self.dataset_path, mode='test')
    
    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory,
            persistent_workers=True if self.num_workers > 0 else False
        )
    
    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )

def train_model_lightning(
    dataset_path: str,
    num_epochs: int = 50,
    batch_size: int = 8,
    learning_rate: float = 1e-3,
    acceleration: int = 4,
    gpus: int = 1,
    use_wandb: bool = True,
    project_name: str = "mri-reconstruction",
    run_name: str = "adaptive-unet-lightning",
    val_split: float = 0.2
):
    """Train model using PyTorch Lightning."""
    
    # Create data module
    data_module = MRIDataModule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=4,
        val_split=val_split
    )
    
    # Create model
    model = MRIReconstructionLightning(
        in_channels=2,
        out_channels=2,
        acceleration=acceleration,
        learning_rate=learning_rate,
    )
    
    # Callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor='val_loss',
            dirpath='checkpoints',
            filename='mri-recon-{epoch:02d}-{val_loss:.4f}',
            save_top_k=3,
            mode='min',
            save_last=True
        ),
        LearningRateMonitor(logging_interval='epoch'),
        EarlyStopping(
            monitor='val_loss',
            patience=10,
            mode='min',
            verbose=True
        )
    ]
    
    # Logger
    logger = None
    if use_wandb:
        logger = WandbLogger(
            project=project_name,
            name=run_name,
            save_dir='logs'
        )
        logger.watch(model)
    
    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator='gpu' if gpus > 0 else 'cpu',
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        precision='16-mixed' if gpus > 0 else 32  # Mixed precision if using GPU
    )
    
    # Train
    trainer.fit(model, data_module)
    
    # Test
    trainer.test(model, data_module)
    
    return model, trainer

def visualize_predictions(model, data_module, num_samples=4):
    """Visualize model predictions vs ground truth."""
    
    # Set model to evaluation mode
    model.eval()
    
    # Get test dataloader
    test_loader = data_module.test_dataloader()
    
    # Get a batch of test data
    batch = next(iter(test_loader))
    kspace_us, kspace_full, mask = batch
    
    # Move to device if using GPU
    device = next(model.parameters()).device
    kspace_us = kspace_us.to(device)
    kspace_full = kspace_full.to(device)
    mask = mask.to(device)
    
    # Make predictions
    with torch.no_grad():        
        # Prediction with built-in data consistency (mask always passed)
        pred_kspace_dc = model(kspace_us, mask)
    
    # Convert k-space to images for visualization
    def kspace_to_image(kspace_tensor) -> np.ndarray:
        """Reconstruct image from k-space using inverse 2D FFT."""
        complex_kspace = torch.complex(kspace_tensor[:, 0], kspace_tensor[:, 1])
        # Inverse FFT to get image
        return np.abs(fftshift(ifft2(ifftshift(complex_kspace))))
    
    # Convert to images
    target_images = kspace_to_image(kspace_full.cpu())
    pred_images = kspace_to_image(pred_kspace_dc.cpu())
    input_images = kspace_to_image(kspace_us.cpu())
    
    # Plot results
    _, axes = plt.subplots(num_samples, 4, figsize=(16, 4*num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)
    
    for i in range(min(num_samples, kspace_us.shape[0])):
        # Input (undersampled)
        axes[i, 0].imshow(input_images[i], cmap='gray')
        axes[i, 0].set_title(f'Input (Undersampled)')
        axes[i, 0].axis('off')
        
        # Prediction
        axes[i, 1].imshow(pred_images[i], cmap='gray')
        axes[i, 1].set_title(f'Prediction')
        axes[i, 1].axis('off')
        
        # Ground Truth
        axes[i, 2].imshow(target_images[i], cmap='gray')
        axes[i, 2].set_title(f'Ground Truth')
        axes[i, 2].axis('off')
        
        # Difference
        diff = np.abs(pred_images[i] - target_images[i])
        im = axes[i, 3].imshow(diff, cmap='hot')
        axes[i, 3].set_title(f'Absolute Difference')
        axes[i, 3].axis('off')
        plt.colorbar(im, ax=axes[i, 3], fraction=0.046, pad=0.04)
        
        # Calculate metrics for this sample
        mse = np.mean((pred_images[i] - target_images[i]) ** 2)
        psnr = 10 * np.log10(np.max(target_images[i]) ** 2 / (mse + 1e-8))
        print(f"Sample {i+1}: MSE = {mse:.6f}, PSNR = {psnr:.2f} dB")
    
    plt.tight_layout()
    
    # Save the figure before showing
    plt.savefig('mri_predictions.png', dpi=300, bbox_inches='tight', facecolor='white')
    
    plt.show()
    
    print(f"📊 Visualization saved as 'mri_predictions.png'")
    
    return pred_images, target_images, input_images

def print_model_summary(model, trainer):
    """Print training summary and final metrics."""
    
    print("\n" + "="*50)
    print("TRAINING SUMMARY")
    print("="*50)
    
    # Get final metrics from trainer
    if trainer.callback_metrics:
        print("\nFinal Metrics:")
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.6f}")
            else:
                print(f"  {key}: {value}")
    
    # Model info
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"\nModel Information:")
    print(f"  Total parameters: {total_params:,}")
    print(f"  Trainable parameters: {trainable_params:,}")
    print(f"  Acceleration factor: {model.hparams.acceleration}")
    print(f"  Learning rate: {model.hparams.learning_rate}")
    
    print("\n" + "="*50)


def load_model(model_path: str, device: str = 'cuda' if torch.cuda.is_available() else 'cpu'):
    """Load the saved model."""
    
    # Load the saved model data with weights_only=False to handle Lightning's AttributeDict
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    
    # Extract hyperparameters
    hparams = checkpoint['hparams']
    
    # Create model instance
    model = AdaptiveUNet(
        in_channels=hparams.get('in_channels', 2),
        out_channels=hparams.get('out_channels', 2),
        acceleration=hparams.get('acceleration', 4)
    )
    
    # Load the state dict
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Move to device and set to eval mode
    model.to(device)
    model.eval()
    
    print(f"Model loaded successfully from {model_path}")
    print(f"Device: {device}")
    print(f"Acceleration factor: {hparams.get('acceleration', 4)}")
    
    return model, hparams


# %%
if __name__ == "__main__":
    # Set random seed for reproducibility
    train = False
    pl.seed_everything(42)
    wandb.login(key="8709b844d342b1f107a01f58f5c666423f2f9656")
    
    if train:
        # Train model
        model, trainer = train_model_lightning(
            dataset_path='./mri_dataset.h5',
            num_epochs=10,
            batch_size=8,
            learning_rate=1e-3,
            acceleration=4,
            gpus=1 if torch.cuda.is_available() else 0,
            use_wandb=True,
            project_name="mri-reconstruction",
            run_name="adaptive-unet",
            val_split=0.1
        )

        # Print training summary
        print_model_summary(model, trainer)

        # Save final model in standard PyTorch format for compatibility
        torch.save({
            'model_state_dict': model.model.state_dict(),
            'hparams': model.hparams
        }, './adaptive-unet.pth')

        print("\nTraining completed! Model saved as 'adaptive-unet.pth'")
    else:
        model, _ = load_model("adaptive-unet.pth")
    
    # Create data module for visualization
    data_module = MRIDataModule(
        dataset_path='./mri_dataset.h5',
        batch_size=16,
        num_workers=8,
        val_split=0.1
    )
    data_module.setup('test')
    
    # Visualize predictions
    print("\nGenerating predictions visualization...")
    pred_images, target_images, input_images = visualize_predictions(
        model, data_module, num_samples=4
    )
    
# %%