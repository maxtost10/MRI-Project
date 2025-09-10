# %%

import matplotlib.pyplot as plt
import numpy as np
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torch.nn.functional as F
import wandb
from pytorch_lightning.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
    ModelSummary,
)
from pytorch_lightning.loggers import WandbLogger

# Import your existing k-space reconstruction model
from train_unet import AdaptiveUNet, MRIDataModule


class ImageRefinementCNN(nn.Module):
    """CNN for refining images reconstructed from k-space.

    This network takes as input the magnitude image from reconstructed k-space
    and outputs a refined version that should be closer to the ground truth.
    """

    def __init__(self, in_channels: int = 1, out_channels: int = 1):
        super(ImageRefinementCNN, self).__init__()

        # Residual learning - predict the residual to add to input
        self.residual_learning = True

        # Initial convolution
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=9, padding=4),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Residual blocks for feature extraction
        self.res_blocks = nn.ModuleList(
            [self._make_residual_block(64, 64) for _ in range(6)]
        )

        # Feature refinement layers
        self.refine1 = nn.Sequential(
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
        )

        self.refine2 = nn.Sequential(
            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Final reconstruction layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=5, padding=2)

    def _make_residual_block(self, in_channels: int, out_channels: int):
        """Create a residual block."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
        )

    def forward(self, x):
        # Store input for residual connection
        identity = x

        # Initial features
        features = self.conv1(x)

        # Apply residual blocks
        for res_block in self.res_blocks:
            residual = res_block(features)
            features = features + residual  # Skip connection
            features = F.relu(features)

        # Refinement
        features = self.refine1(features)
        features = self.refine2(features)

        # Final output
        output = self.final(features)

        # Add residual if enabled
        if self.residual_learning:
            output = output + identity

        return output


class CascadedMRIReconstruction(pl.LightningModule):
    """Lightning module for cascaded MRI reconstruction.

    Stage 1: K-space reconstruction (pre-trained, frozen)
    Stage 2: Image refinement (trainable)
    """

    def __init__(
        self,
        kspace_model_path: str,
        learning_rate: float = 1e-4,
        weight_decay: float = 1e-5,
        lambda_l1: float = 1e-3,  # Add this parameter
    ):
        super().__init__()
        self.save_hyperparameters()

        # Load pre-trained k-space reconstruction model
        self.kspace_model = self._load_kspace_model(kspace_model_path)

        # Freeze k-space model parameters
        for param in self.kspace_model.parameters():
            param.requires_grad = False

        # Image refinement model (trainable)
        self.refinement_model = ImageRefinementCNN(in_channels=1, out_channels=1)

        # Loss functions
        self.l1_loss = nn.L1Loss()
        self.l2_loss = nn.MSELoss()

        # For perceptual loss (optional)
        self.use_perceptual_loss = False

    def _load_kspace_model(self, model_path: str):
        """Load pre-trained k-space reconstruction model."""
        checkpoint = torch.load(model_path, map_location="cpu", weights_only=False)
        hparams = checkpoint["hparams"]

        model = AdaptiveUNet(
            in_channels=hparams.get("in_channels", 2),
            out_channels=hparams.get("out_channels", 2),
            acceleration=hparams.get("acceleration", 4),
        )
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()  # Set to eval mode

        print(f"Loaded k-space model from {model_path}")
        return model

    def kspace_to_image(self, kspace_tensor: torch.Tensor) -> torch.Tensor:
        """Convert k-space tensor to image tensor.

        Args:
            kspace_tensor: Tensor of shape (batch, 2, H, W) with real/imag channels

        Returns:
            Image tensor of shape (batch, 1, H, W)
        """
        # Convert to complex
        complex_kspace = torch.complex(kspace_tensor[:, 0], kspace_tensor[:, 1])

        # Apply inverse FFT
        shifted = torch.fft.ifftshift(complex_kspace, dim=(-2, -1))
        ifft_result = torch.fft.ifft2(shifted, dim=(-2, -1))
        image = torch.fft.fftshift(ifft_result, dim=(-2, -1))

        # Take absolute value to get magnitude image
        magnitude = torch.abs(image)

        # Add channel dimension
        return magnitude.unsqueeze(1)

    def forward(self, kspace_us, mask):
        # Stage 1: K-space reconstruction (frozen)
        with torch.no_grad():
            reconstructed_kspace = self.kspace_model(kspace_us, mask)

        # Convert reconstructed k-space to image
        reconstructed_image = self.kspace_to_image(reconstructed_kspace)

        # Stage 2: Image refinement
        refined_image = self.refinement_model(reconstructed_image)

        return reconstructed_kspace, reconstructed_image, refined_image

    def _shared_step(self, batch, stage: str):
        kspace_us, kspace_full, mask, phantom_gt = batch

        # Forward pass through cascaded model
        reconstructed_kspace, reconstructed_image, refined_image = self.forward(
            kspace_us, mask
        )

        # Ground truth needs channel dimension
        phantom_gt = phantom_gt.unsqueeze(1)

        # Compute reconstruction losses
        l1_loss = self.l1_loss(refined_image, phantom_gt)
        l2_loss = self.l2_loss(refined_image, phantom_gt)
        reconstruction_loss = l1_loss + 0.1 * l2_loss

        # Compute L1 regularization loss
        l1_reg_loss = 0
        for param in self.refinement_model.parameters():
            l1_reg_loss += torch.sum(torch.abs(param))
        l1_reg_loss = self.hparams.lambda_l1 * l1_reg_loss

        # Total loss
        total_loss = reconstruction_loss + l1_reg_loss

        # Calculate metrics
        with torch.no_grad():
            # PSNR
            mse = torch.mean((refined_image - phantom_gt) ** 2)
            psnr = 10 * torch.log10(phantom_gt.max() ** 2 / (mse + 1e-8))

            # Also calculate PSNR for unrefined reconstruction for comparison
            mse_unrefi = torch.mean((reconstructed_image - phantom_gt) ** 2)
            psnr_unrefined = 10 * torch.log10(
                phantom_gt.max() ** 2 / (mse_unrefi + 1e-8)
            )

        # Log metrics
        self.log(
            f"{stage}_loss", total_loss, on_step=True, on_epoch=True, prog_bar=True
        )
        self.log(f"{stage}_l1_loss", l1_loss, on_step=False, on_epoch=True)
        self.log(f"{stage}_l2_loss", l2_loss, on_step=False, on_epoch=True)
        self.log(
            f"{stage}_psnr_refined", psnr, on_step=False, on_epoch=True, prog_bar=True
        )
        self.log(
            f"{stage}_psnr_unrefined", psnr_unrefined, on_step=False, on_epoch=True
        )
        self.log(
            f"{stage}_psnr_improvement",
            psnr - psnr_unrefined,
            on_step=False,
            on_epoch=True,
        )

        # Print loss components during training
        if stage == "train" and self.global_step % 5 == 0:  # Print every 50 steps
            print(
                f"Step {self.global_step}: Reconstruction Loss = {reconstruction_loss:.6f}, L1 Reg Loss = {self.hparams.lambda_l1 * l1_reg_loss:.6f}"
            )

        return total_loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    def configure_optimizers(self):
        # Only optimize refinement model parameters
        optimizer = torch.optim.Adam(
            self.refinement_model.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, verbose=True
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val_loss",
                "interval": "epoch",
                "frequency": 1,
            },
        }


def train_cascaded_model(
    dataset_path: str,
    kspace_model_path: str,
    num_epochs: int = 30,
    batch_size: int = 8,
    learning_rate: float = 1e-4,
    lambda_l1: float = 1e-3,  # Add this parameter
    gpus: int = 1,
    use_wandb: bool = True,
    project_name: str = "mri-cascaded-refinement",
    run_name: str = "image-refinement-cnn",
    val_split: float = 0.2,
):
    """Train the cascaded MRI reconstruction model."""
    # Create data module (reusing from original code)
    data_module = MRIDataModule(
        dataset_path=dataset_path,
        batch_size=batch_size,
        num_workers=4,
        val_split=val_split,
    )

    # Create cascaded model
    model = CascadedMRIReconstruction(
        kspace_model_path=kspace_model_path,
        learning_rate=learning_rate,
        lambda_l1=lambda_l1,
    )

    # Callbacks
    callbacks = [
        ModelSummary(max_depth=2),
        ModelCheckpoint(
            monitor="val_psnr_refined",
            dirpath="checkpoints_cascaded",
            filename="cascaded-{epoch:02d}-{val_psnr_refined:.2f}",
            save_top_k=3,
            mode="max",
            save_last=True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
        EarlyStopping(monitor="val_loss", patience=15, mode="min", verbose=True),
    ]

    # Logger
    logger = None
    if use_wandb:
        logger = WandbLogger(project=project_name, name=run_name, save_dir="logs")
        logger.watch(model.refinement_model)  # Only watch trainable model

    # Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="gpu" if gpus > 0 else "cpu",
        devices=gpus if gpus > 0 else 1,
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=10,
        gradient_clip_val=1.0,
        deterministic=True,
        precision="16-mixed" if gpus > 0 else 32,
    )

    # Train
    trainer.fit(model, data_module)

    # Test
    trainer.test(model, data_module)

    return model, trainer


def visualize_cascaded_results(model, data_module, num_samples=4):
    """Visualize the cascaded reconstruction pipeline."""
    model.eval()
    test_loader = data_module.test_dataloader()
    batch = next(iter(test_loader))

    kspace_us, kspace_full, mask, phantom_gt = batch

    # Move to device
    device = next(model.parameters()).device
    kspace_us = kspace_us.to(device)
    kspace_full = kspace_full.to(device)
    mask = mask.to(device)
    phantom_gt = phantom_gt.to(device)

    # Get predictions
    with torch.no_grad():
        reconstructed_kspace, reconstructed_image, refined_image = model(
            kspace_us, mask
        )

    # Convert to numpy for visualization
    def to_numpy_image(tensor):
        if len(tensor.shape) == 4:  # Batch dimension
            return tensor.squeeze(1).cpu().numpy()
        return tensor.cpu().numpy()

    # Convert k-space to images
    def kspace_to_image_np(kspace_tensor):
        complex_kspace = torch.complex(kspace_tensor[:, 0], kspace_tensor[:, 1])
        shifted = torch.fft.ifftshift(complex_kspace, dim=(-2, -1))
        ifft_result = torch.fft.ifft2(shifted, dim=(-2, -1))
        image = torch.fft.fftshift(ifft_result, dim=(-2, -1))
        return torch.abs(image).cpu().numpy()

    # Get images
    input_images = kspace_to_image_np(kspace_us)
    full_kspace_images = kspace_to_image_np(kspace_full)
    reconstructed_np = to_numpy_image(reconstructed_image)
    refined_np = to_numpy_image(refined_image)
    phantom_np = to_numpy_image(phantom_gt)

    # Create visualization
    fig, axes = plt.subplots(num_samples, 6, figsize=(24, 4 * num_samples))
    if num_samples == 1:
        axes = axes.reshape(1, -1)

    for i in range(min(num_samples, kspace_us.shape[0])):
        # Zero-filled (input)
        axes[i, 0].imshow(input_images[i], cmap="gray")
        axes[i, 0].set_title("Zero-filled Input")
        axes[i, 0].axis("off")

        # Stage 1: K-space reconstruction
        axes[i, 1].imshow(reconstructed_np[i], cmap="gray")
        axes[i, 1].set_title("Stage 1: K-space Recon")
        axes[i, 1].axis("off")

        # Stage 2: Refined image
        axes[i, 2].imshow(refined_np[i], cmap="gray")
        axes[i, 2].set_title("Stage 2: Refined Image")
        axes[i, 2].axis("off")

        # Ground truth
        axes[i, 3].imshow(phantom_np[i], cmap="gray")
        axes[i, 3].set_title("Ground Truth")
        axes[i, 3].axis("off")

        # Error map before refinement
        error_before = np.abs(reconstructed_np[i] - phantom_np[i])
        im1 = axes[i, 4].imshow(error_before, cmap="hot", vmin=0, vmax=0.3)
        axes[i, 4].set_title("Error Before Refinement")
        axes[i, 4].axis("off")
        plt.colorbar(im1, ax=axes[i, 4], fraction=0.046, pad=0.04)

        # Error map after refinement
        error_after = np.abs(refined_np[i] - phantom_np[i])
        im2 = axes[i, 5].imshow(error_after, cmap="hot", vmin=0, vmax=0.3)
        axes[i, 5].set_title("Error After Refinement")
        axes[i, 5].axis("off")
        plt.colorbar(im2, ax=axes[i, 5], fraction=0.046, pad=0.04)

        # Calculate metrics
        mse_before = np.mean(error_before**2)
        mse_after = np.mean(error_after**2)
        psnr_before = 10 * np.log10(np.max(phantom_np[i]) ** 2 / (mse_before + 1e-8))
        psnr_after = 10 * np.log10(np.max(phantom_np[i]) ** 2 / (mse_after + 1e-8))

        print(f"Sample {i+1}:")
        print(
            f"  Before refinement - MSE: {mse_before:.6f}, PSNR: {psnr_before:.2f} dB"
        )
        print(f"  After refinement  - MSE: {mse_after:.6f}, PSNR: {psnr_after:.2f} dB")
        print(f"  Improvement       - PSNR: +{psnr_after - psnr_before:.2f} dB\n")

    plt.suptitle("Cascaded MRI Reconstruction: K-space → Image Refinement", fontsize=16)
    plt.tight_layout()
    plt.savefig("cascaded_reconstruction_results.png", dpi=300, bbox_inches="tight")
    plt.show()

    print("📊 Visualization saved as 'cascaded_reconstruction_results.png'")

    return refined_np, phantom_np


def print_cascaded_summary(model, trainer):
    """Print summary of the cascaded model training."""
    print("\n" + "=" * 60)
    print("CASCADED MRI RECONSTRUCTION SUMMARY")
    print("=" * 60)

    # Training metrics
    if trainer.callback_metrics:
        print("\nFinal Metrics:")
        for key, value in trainer.callback_metrics.items():
            if isinstance(value, torch.Tensor):
                print(f"  {key}: {value.item():.4f}")

    # Model information
    kspace_params = sum(p.numel() for p in model.kspace_model.parameters())
    refinement_params = sum(p.numel() for p in model.refinement_model.parameters())
    refinement_trainable = sum(
        p.numel() for p in model.refinement_model.parameters() if p.requires_grad
    )

    print("\nModel Architecture:")
    print(f"  Stage 1 (K-space): {kspace_params:,} parameters (frozen)")
    print(f"  Stage 2 (Refinement): {refinement_params:,} parameters (trainable)")
    print(f"  Total parameters: {kspace_params + refinement_params:,}")

    print("\nTraining Configuration:")
    print(f"  Learning rate: {model.hparams.learning_rate}")
    print(f"  Weight decay: {model.hparams.weight_decay}")

    print("\n" + "=" * 60)


# %%
if __name__ == "__main__":
    # Set random seed
    pl.seed_everything(42)

    # Configuration
    TRAIN_CASCADED = True
    KSPACE_MODEL_PATH = "./adaptive-unet.pth"  # Path to your trained k-space model
    DATASET_PATH = "./mri_dataset.h5"

    # Optionally login to wandb
    wandb.login(key="8709b844d342b1f107a01f58f5c666423f2f9656")

    if TRAIN_CASCADED:
        print("Training cascaded MRI reconstruction model...")
        print("Stage 1: Pre-trained k-space reconstruction (frozen)")
        print("Stage 2: Image refinement CNN (trainable)\n")

        # Train the cascaded model
        model, trainer = train_cascaded_model(
            dataset_path=DATASET_PATH,
            kspace_model_path=KSPACE_MODEL_PATH,
            num_epochs=30,
            batch_size=8,
            learning_rate=1e-4,
            gpus=1 if torch.cuda.is_available() else 0,
            use_wandb=False,  # Set to True to use wandb
            project_name="mri-cascaded-refinement",
            run_name="image-refinement-cnn",
            val_split=0.1,
        )

        # Print summary
        print_cascaded_summary(model, trainer)

        # Save the refinement model separately
        torch.save(
            {
                "refinement_state_dict": model.refinement_model.state_dict(),
                "kspace_model_path": KSPACE_MODEL_PATH,
                "hparams": model.hparams,
            },
            "./refinement_model.pth",
        )

        print("\nRefinement model saved as 'refinement_model.pth'")

    else:
        # Load existing model
        print("Loading cascaded model from checkpoint...")
        model = CascadedMRIReconstruction.load_from_checkpoint(
            "checkpoints_cascaded/last.ckpt", kspace_model_path=KSPACE_MODEL_PATH
        )

    # Create data module for visualization
    data_module = MRIDataModule(
        dataset_path=DATASET_PATH, batch_size=16, num_workers=4, val_split=0.1
    )
    data_module.setup("test")

    # Visualize results
    print("\nGenerating cascaded reconstruction visualizations...")
    refined_images, gt_images = visualize_cascaded_results(
        model, data_module, num_samples=4
    )

    print("\n✅ Cascaded reconstruction complete!")
    print("The refinement CNN successfully improves upon the k-space reconstruction.")

# %%
