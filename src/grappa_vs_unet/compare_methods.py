# %%
import torch
import torch.nn as nn
import numpy as np
import h5py
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Tuple
from tqdm import tqdm
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import pandas as pd
from numpy.fft import ifft2, fftshift, ifftshift

# Import from your existing script
from train_unet import AdaptiveUNet

def complex_to_tensor(complex_array: np.ndarray) -> torch.Tensor:
    """Convert complex numpy array to torch tensor with real/imag channels."""
    return torch.stack([
        torch.from_numpy(np.real(complex_array)).float(),
        torch.from_numpy(np.imag(complex_array)).float()
    ])

def tensor_to_complex(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor with real/imag channels to complex numpy array."""
    return tensor[0].numpy() + 1j * tensor[1].numpy()

def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Reconstruct image from k-space using inverse 2D FFT."""
    return np.abs(fftshift(ifft2(ifftshift(kspace))))

class GRAPPA:
    """GRAPPA reconstruction for single-coil data (simplified version)."""
    
    def __init__(self, kernel_size: Tuple[int, int] = (5, 5)):
        self.kernel_size = kernel_size
        self.weights = None
    
    def calibrate(self, kspace: np.ndarray, mask: np.ndarray):
        """Calibrate GRAPPA weights using ACS region."""
        ny, nx = kspace.shape
        ky, kx = self.kernel_size
        
        # Find ACS region (consecutive fully sampled lines)
        fully_sampled = np.all(mask, axis=1)
        acs_indices = np.where(fully_sampled)[0]
        
        if len(acs_indices) < ky:
            raise ValueError("Not enough ACS lines for calibration")
        
        # Prepare calibration data
        sources = []
        targets = []
        
        # For each point in ACS region
        acs_start, acs_end = acs_indices[0], acs_indices[-1] + 1
        
        for target_y in range(acs_start + ky//2, acs_end - ky//2):
            for target_x in range(kx//2, nx - kx//2):
                # Extract source patch (excluding center point)
                source_patch = []
                for dy in range(-ky//2, ky//2 + 1):
                    for dx in range(-kx//2, kx//2 + 1):
                        if dy != 0 or dx != 0:  # Exclude target point
                            source_patch.append(kspace[target_y + dy, target_x + dx])
                
                sources.append(source_patch)
                targets.append(kspace[target_y, target_x])
        
        sources = np.array(sources)
        targets = np.array(targets)
        
        # Solve for weights using least squares
        # Separate real and imaginary parts
        sources_real = np.real(sources)
        sources_imag = np.imag(sources)
        targets_real = np.real(targets)
        targets_imag = np.imag(targets)
        
        # Stack real and imaginary parts
        sources_combined = np.hstack([sources_real, sources_imag])
        
        # Solve for real and imaginary weights separately
        weights_real = np.linalg.lstsq(sources_combined, targets_real, rcond=None)[0]
        weights_imag = np.linalg.lstsq(sources_combined, targets_imag, rcond=None)[0]
        
        self.weights = (weights_real, weights_imag)
    
    def reconstruct(self, kspace_undersampled: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """Reconstruct missing k-space points using GRAPPA."""
        if self.weights is None:
            raise ValueError("GRAPPA weights not calibrated")
        
        ny, nx = kspace_undersampled.shape
        ky, kx = self.kernel_size
        kspace_filled = kspace_undersampled.copy()
        weights_real, weights_imag = self.weights
        
        # Find missing points
        for y in range(ky//2, ny - ky//2):
            for x in range(kx//2, nx - kx//2):
                if not mask[y, x]:  # Missing point
                    # Extract source patch
                    source_patch = []
                    for dy in range(-ky//2, ky//2 + 1):
                        for dx in range(-kx//2, kx//2 + 1):
                            if dy != 0 or dx != 0:
                                source_patch.append(kspace_filled[y + dy, x + dx])
                    
                    source_patch = np.array(source_patch)
                    
                    # Apply weights
                    source_combined = np.hstack([np.real(source_patch), np.imag(source_patch)])
                    pred_real = np.dot(source_combined, weights_real)
                    pred_imag = np.dot(source_combined, weights_imag)
                    
                    kspace_filled[y, x] = pred_real + 1j * pred_imag
        
        return kspace_filled

def evaluate_reconstruction(ground_truth: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
    """Calculate reconstruction metrics."""
    # Ensure same scale
    gt_normalized = ground_truth / np.max(ground_truth)
    recon_normalized = reconstruction / np.max(reconstruction)
    
    metrics = {
        'PSNR': psnr(gt_normalized, recon_normalized, data_range=1.0),
        'SSIM': ssim(gt_normalized, recon_normalized, data_range=1.0),
        'RMSE': np.sqrt(np.mean((gt_normalized - recon_normalized)**2))
    }
    
    return metrics

def compare_methods(
        dataset_path: str,
        model_path: str,
        num_samples: int = 20,
        device: str = 'cuda' if torch.cuda.is_available() else 'cpu'
    ):
    """Compare U-Net and GRAPPA reconstruction methods."""
    
    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hparams = checkpoint['hparams']
    
    model = AdaptiveUNet(
        in_channels=hparams.get('in_channels', 2),
        out_channels=hparams.get('out_channels', 2),
        acceleration=hparams.get('acceleration', 4)
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Initialize results storage
    results = {
        'sample_idx': [],
        'method': [],
        'PSNR': [],
        'SSIM': [],
        'RMSE': []
    }
    
    # Load test data
    with h5py.File(dataset_path, 'r') as f:
        test_data = f['test']
        
        # Process samples
        for idx in tqdm(range(min(num_samples, test_data['phantoms'].shape[0])), desc="Processing samples"):
            # Load data
            kspace_full = test_data['kspace_full'][idx]
            kspace_undersampled = test_data['kspace_undersampled'][idx]
            mask = test_data['masks'][idx]
            
            # Ground truth reconstruction
            gt_image = kspace_to_image(kspace_full)
            
            # Zero-filled reconstruction
            zerofilled_image = kspace_to_image(kspace_undersampled)
            zerofilled_metrics = evaluate_reconstruction(gt_image, zerofilled_image)
            
            results['sample_idx'].append(idx)
            results['method'].append('Zero-filled')
            results['PSNR'].append(zerofilled_metrics['PSNR'])
            results['SSIM'].append(zerofilled_metrics['SSIM'])
            results['RMSE'].append(zerofilled_metrics['RMSE'])
            
            # U-Net reconstruction
            with torch.no_grad():
                kspace_tensor = complex_to_tensor(kspace_undersampled).unsqueeze(0).to(device)
                mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
                
                pred_kspace = model(kspace_tensor, mask_tensor)
                pred_kspace_np = tensor_to_complex(pred_kspace[0].cpu())
                unet_image = kspace_to_image(pred_kspace_np)
            
            unet_metrics = evaluate_reconstruction(gt_image, unet_image)
            
            results['sample_idx'].append(idx)
            results['method'].append('U-Net')
            results['PSNR'].append(unet_metrics['PSNR'])
            results['SSIM'].append(unet_metrics['SSIM'])
            results['RMSE'].append(unet_metrics['RMSE'])
            
            # GRAPPA reconstruction
            try:
                grappa = GRAPPA(kernel_size=(5, 5))
                grappa.calibrate(kspace_undersampled, mask)
                kspace_grappa = grappa.reconstruct(kspace_undersampled, mask)
                grappa_image = kspace_to_image(kspace_grappa)
                
                grappa_metrics = evaluate_reconstruction(gt_image, grappa_image)
                
                results['sample_idx'].append(idx)
                results['method'].append('GRAPPA')
                results['PSNR'].append(grappa_metrics['PSNR'])
                results['SSIM'].append(grappa_metrics['SSIM'])
                results['RMSE'].append(grappa_metrics['RMSE'])
                
            except Exception as e:
                print(f"GRAPPA failed for sample {idx}: {e}")
                results['sample_idx'].append(idx)
                results['method'].append('GRAPPA')
                results['PSNR'].append(np.nan)
                results['SSIM'].append(np.nan)
                results['RMSE'].append(np.nan)
            
            # Visualize first few samples
            if idx < 5:
                _, axes = plt.subplots(2, 4, figsize=(16, 8))
                
                # Images
                axes[0, 0].imshow(gt_image, cmap='gray')
                axes[0, 0].set_title('Ground Truth')
                axes[0, 0].axis('off')
                
                axes[0, 1].imshow(zerofilled_image, cmap='gray')
                axes[0, 1].set_title(f'Zero-filled\nPSNR: {zerofilled_metrics["PSNR"]:.2f}')
                axes[0, 1].axis('off')
                
                axes[0, 2].imshow(unet_image, cmap='gray')
                axes[0, 2].set_title(f'U-Net\nPSNR: {unet_metrics["PSNR"]:.2f}')
                axes[0, 2].axis('off')
                
                if not np.isnan(results['PSNR'][-1]):
                    axes[0, 3].imshow(grappa_image, cmap='gray')
                    axes[0, 3].set_title(f'GRAPPA\nPSNR: {grappa_metrics["PSNR"]:.2f}')
                else:
                    axes[0, 3].text(0.5, 0.5, 'GRAPPA Failed', ha='center', va='center')
                axes[0, 3].axis('off')
                
                # Error maps
                axes[1, 0].imshow(mask, cmap='gray')
                axes[1, 0].set_title('Sampling Mask')
                axes[1, 0].axis('off')
                
                error_zero = np.abs(gt_image - zerofilled_image)
                axes[1, 1].imshow(error_zero, cmap='hot', vmin=0, vmax=0.5)
                axes[1, 1].set_title('Zero-filled Error')
                axes[1, 1].axis('off')
                
                error_unet = np.abs(gt_image - unet_image)
                axes[1, 2].imshow(error_unet, cmap='hot', vmin=0, vmax=0.5)
                axes[1, 2].set_title('U-Net Error')
                axes[1, 2].axis('off')
                
                if not np.isnan(results['PSNR'][-1]):
                    error_grappa = np.abs(gt_image - grappa_image)
                    axes[1, 3].imshow(error_grappa, cmap='hot', vmin=0, vmax=0.5)
                    axes[1, 3].set_title('GRAPPA Error')
                axes[1, 3].axis('off')
                
                plt.suptitle(f'Sample {idx} Comparison')
                plt.tight_layout()
                plt.savefig(f'comparison_sample_{idx}.png', dpi=150)
                plt.show()
    
    # Create results dataframe
    results_df = pd.DataFrame(results)
    
    # Summary statistics
    print("\n=== Summary Statistics ===")
    summary = results_df.groupby('method')[['PSNR', 'SSIM', 'RMSE']].agg(['mean', 'std'])
    print(summary)
    
    # Box plots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    metrics = ['PSNR', 'SSIM', 'RMSE']
    for i, metric in enumerate(metrics):
        data_to_plot = [results_df[results_df['method'] == method][metric].dropna() 
                       for method in ['Zero-filled', 'U-Net', 'GRAPPA']]
        
        axes[i].boxplot(data_to_plot, labels=['Zero-filled', 'U-Net', 'GRAPPA'])
        axes[i].set_title(f'{metric} Distribution')
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
    
    plt.suptitle('Reconstruction Performance Comparison')
    plt.tight_layout()
    plt.savefig('performance_comparison.png', dpi=150)
    plt.show()
    
    # Save detailed results
    results_df.to_csv('reconstruction_results.csv', index=False)
    print("\nDetailed results saved to 'reconstruction_results.csv'")
    
    return results_df

# %%
if __name__ == "__main__":
    # Run comparison
    results = compare_methods(
        dataset_path='./mri_dataset.h5',
        model_path='./adaptive-unet.pth',
        num_samples=20
    )
    
    # Additional analysis
    print("\n=== Performance Improvement over Zero-filled ===")
    zero_filled_psnr = results[results['method'] == 'Zero-filled']['PSNR'].mean()
    unet_psnr = results[results['method'] == 'U-Net']['PSNR'].mean()
    grappa_psnr = results[results['method'] == 'GRAPPA']['PSNR'].mean()
    
    print(f"U-Net improvement: {unet_psnr - zero_filled_psnr:.2f} dB")
    print(f"GRAPPA improvement: {grappa_psnr - zero_filled_psnr:.2f} dB")

# %%