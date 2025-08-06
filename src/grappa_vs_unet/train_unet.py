# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import h5py
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
from typing import Tuple
import wandb

# Initialize wandb (optional, comment out if not using)
# wandb.init(project="mri-reconstruction", name="unet-kspace")

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

class UNet(nn.Module):
    """U-Net architecture for k-space reconstruction."""
    
    def __init__(self, in_channels: int = 2, out_channels: int = 2):
        super(UNet, self).__init__()
        
        # Encoder
        self.enc1 = self.conv_block(in_channels, 64)
        self.enc2 = self.conv_block(64, 128)
        self.enc3 = self.conv_block(128, 256)
        self.enc4 = self.conv_block(256, 512)
        
        # Bottleneck
        self.bottleneck = self.conv_block(512, 1024)
        
        # Decoder
        self.upconv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dec4 = self.conv_block(1024, 512)
        
        self.upconv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dec3 = self.conv_block(512, 256)
        
        self.upconv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dec2 = self.conv_block(256, 128)
        
        self.upconv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dec1 = self.conv_block(128, 64)
        
        # Final layer
        self.final = nn.Conv2d(64, out_channels, kernel_size=1)
        
        self.pool = nn.MaxPool2d(2)
    
    def conv_block(self, in_channels: int, out_channels: int):
        """Convolutional block with batch norm and ReLU."""
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )
    
    def forward(self, x):
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
        
        return self.final(dec1)

class DataConsistencyLayer(nn.Module):
    """Enforce data consistency in k-space."""
    
    def forward(self, pred_kspace: torch.Tensor, 
                undersampled_kspace: torch.Tensor, 
                mask: torch.Tensor) -> torch.Tensor:
        """
        Replace predicted k-space values with acquired values where available.
        
        Args:
            pred_kspace: Predicted k-space (B, 2, H, W) - real and imag channels
            undersampled_kspace: Original undersampled k-space (B, 2, H, W)
            mask: Sampling mask (B, H, W)
            
        Returns:
            Data-consistent k-space
        """
        # Expand mask to match k-space dimensions
        mask_expanded = mask.unsqueeze(1).expand_as(pred_kspace)
        
        # Replace predicted values with measured values where available
        return pred_kspace * (1 - mask_expanded) + undersampled_kspace * mask_expanded

def train_model(
        dataset_path: str,
        num_epochs: int = 100,
        batch_size: int = 8,
        learning_rate: float = 1e-3,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu',
        save_path: str = './unet_model.pth'
    ):
    """Train U-Net model for MRI reconstruction."""
    
    # Create datasets and dataloaders
    train_dataset = MRIDataset(dataset_path, mode='train')
    val_dataset = MRIDataset(dataset_path, mode='test')
    
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, num_workers=4)
    
    # Initialize model
    model = UNet(in_channels=2, out_channels=2).to(device)
    dc_layer = DataConsistencyLayer()
    
    # Loss and optimizer
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    
    # Training history
    train_losses = []
    val_losses = []
    best_val_loss = float('inf')
    
    print(f"Training on {device}")
    print(f"Train samples: {len(train_dataset)}, Val samples: {len(val_dataset)}")
    
    for epoch in range(num_epochs):
        # Training
        model.train()
        train_loss = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Train]')
        for kspace_us, kspace_full, mask in pbar:
            kspace_us = kspace_us.to(device)
            kspace_full = kspace_full.to(device)
            mask = mask.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            pred_kspace = model(kspace_us)
            
            # Apply data consistency
            pred_kspace = dc_layer(pred_kspace, kspace_us, mask)
            
            # Compute loss
            loss = criterion(pred_kspace, kspace_full)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})
        
        avg_train_loss = train_loss / len(train_loader)
        train_losses.append(avg_train_loss)
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for kspace_us, kspace_full, mask in tqdm(val_loader, desc=f'Epoch {epoch+1}/{num_epochs} [Val]'):
                kspace_us = kspace_us.to(device)
                kspace_full = kspace_full.to(device)
                mask = mask.to(device)
                
                pred_kspace = model(kspace_us)
                pred_kspace = dc_layer(pred_kspace, kspace_us, mask)
                
                loss = criterion(pred_kspace, kspace_full)
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        
        # Learning rate scheduling
        scheduler.step(avg_val_loss)
        
        print(f"Epoch {epoch+1}: Train Loss = {avg_train_loss:.6f}, Val Loss = {avg_val_loss:.6f}")
        
        # Save best model
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, save_path)
            print(f"Model saved!")
        
        # Log to wandb if available
        # wandb.log({
        #     'train_loss': avg_train_loss,
        #     'val_loss': avg_val_loss,
        #     'learning_rate': optimizer.param_groups[0]['lr']
        # })
    
    # Plot training history
    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Train Loss')
    plt.plot(val_losses, label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.title('Training History')
    plt.savefig('training_history.png')
    plt.show()
    
    return model

# %%
if __name__ == "__main__":
    # Train model
    model = train_model(
        dataset_path='./mri_dataset.h5',
        num_epochs=50,
        batch_size=8,
        learning_rate=1e-3,
        save_path='./unet_model.pth'
    )