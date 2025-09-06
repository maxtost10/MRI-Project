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
from numpy.fft import fft2, ifft2, fftshift, ifftshift

# Import from your existing scripts
from train_unet import AdaptiveUNet, MRIDataModule
from train_cascaded import CascadedMRIReconstruction, ImageRefinementCNN


def complex_to_tensor(complex_array: np.ndarray) -> torch.Tensor:
    """Convert complex numpy array to torch tensor with real/imag channels."""
    return torch.stack(
        [
            torch.from_numpy(np.real(complex_array)).float(),
            torch.from_numpy(np.imag(complex_array)).float(),
        ]
    )


def tensor_to_complex(tensor: torch.Tensor) -> np.ndarray:
    """Convert torch tensor with real/imag channels to complex numpy array."""
    return tensor[0].numpy() + 1j * tensor[1].numpy()


def kspace_to_image(kspace: np.ndarray) -> np.ndarray:
    """Reconstruct image from k-space using inverse 2D FFT."""
    return np.abs(fftshift(ifft2(ifftshift(kspace))))


def evaluate_reconstruction(ground_truth: np.ndarray, reconstruction: np.ndarray) -> Dict[str, float]:
    """Evaluate reconstruction quality metrics."""
    # Normalize images to [0, 1] range for consistent metrics
    gt_norm = (ground_truth - ground_truth.min()) / (ground_truth.max() - ground_truth.min())
    rec_norm = (reconstruction - reconstruction.min()) / (reconstruction.max() - reconstruction.min())
    
    psnr_val = psnr(gt_norm, rec_norm, data_range=1.0)
    ssim_val = ssim(gt_norm, rec_norm, data_range=1.0)
    rmse_val = np.sqrt(np.mean((gt_norm - rec_norm) ** 2))
    
    return {
        "PSNR": psnr_val,
        "SSIM": ssim_val,
        "RMSE": rmse_val
    }


class TVDenoising:
    """Simple TV denoising for removing artifacts from zero-filled reconstruction."""
    
    def __init__(self, lambda_tv: float = 0.01, max_iter: int = 70):
        """
        Initialize TV denoising.
        
        Args:
            lambda_tv: Regularization parameter for TV penalty
            max_iter: Maximum number of iterations
        """
        self.lambda_tv = lambda_tv
        self.max_iter = max_iter
        
    def _compute_gradient(self, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Compute image gradients using finite differences."""
        # Gradient in x direction (with circular boundary)
        dx = np.roll(x, -1, axis=1) - x
        # Gradient in y direction (with circular boundary)  
        dy = np.roll(x, -1, axis=0) - x
        return dx, dy
    
    def _compute_divergence(self, px: np.ndarray, py: np.ndarray) -> np.ndarray:
        """Compute divergence (adjoint of gradient operator)."""
        # Divergence is negative adjoint of gradient
        div_x = px - np.roll(px, 1, axis=1)
        div_y = py - np.roll(py, 1, axis=0)
        return div_x + div_y
    
    def _project_to_unit_ball(self, px: np.ndarray, py: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Project onto the L2 ball (dual constraint for TV)."""
        norm = np.sqrt(px**2 + py**2)
        norm = np.maximum(norm, 1.0)
        return px / norm, py / norm
    
    def denoise(self, noisy_image: np.ndarray) -> np.ndarray:
        """
        Apply TV denoising to remove artifacts.
        
        Solves: argmin_u { 0.5 * ||u - noisy_image||^2 + lambda_tv * TV(u) }
        
        Args:
            noisy_image: Input image with artifacts (e.g., zero-filled reconstruction)
            
        Returns:
            Denoised image
        """
        # Initialize dual variables
        px = np.zeros_like(noisy_image)
        py = np.zeros_like(noisy_image)
        
        tau = 0.25  # Step size for dual problem
        
        for _ in range(self.max_iter):
            # Compute divergence
            div_p = self._compute_divergence(px, py)
            
            # Update primal variable: u = x - λ·div(p)
            u = noisy_image + self.lambda_tv * div_p
            
            # Compute gradient of u
            dx, dy = self._compute_gradient(u)
            
            # Update dual variables
            px_new = px + tau * dx
            py_new = py + tau * dy
            
            # Project onto unit ball
            px, py = self._project_to_unit_ball(px_new, py_new)
        
        # Final primal variable
        div_p = self._compute_divergence(px, py)
        return noisy_image + self.lambda_tv * div_p


def compare_methods(
    dataset_path: str,
    cascaded_model_path: str,
    kspace_model_path: str,
    num_samples: int = 20,
    lambda_tv: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Compare Cascaded Model, K-space only U-Net, and TV denoising reconstruction methods."""

    # Load cascaded model
    print("Loading cascaded model...")
    cascaded_model = CascadedMRIReconstruction.load_from_checkpoint(
        cascaded_model_path,
        kspace_model_path=kspace_model_path,
        map_location=device
    )
    cascaded_model.to(device)
    cascaded_model.eval()

    # Load standalone k-space model for comparison
    print("Loading standalone k-space model...")
    checkpoint = torch.load(kspace_model_path, map_location=device, weights_only=False)
    hparams = checkpoint["hparams"]

    kspace_model = AdaptiveUNet(
        in_channels=hparams.get("in_channels", 2),
        out_channels=hparams.get("out_channels", 2),
        acceleration=hparams.get("acceleration", 4),
    )
    kspace_model.load_state_dict(checkpoint["model_state_dict"])
    kspace_model.to(device)
    kspace_model.eval()

    # Initialize TV denoiser
    tv_denoiser = TVDenoising(lambda_tv=lambda_tv, max_iter=70)

    # Initialize results storage
    results = {"sample_idx": [], "method": [], "PSNR": [], "SSIM": [], "RMSE": []}

    # Load test data
    with h5py.File(dataset_path, "r") as f:
        test_data = f["test"]

        # Process samples
        for idx in tqdm(
            range(min(num_samples, test_data["phantoms"].shape[0])),
            desc="Processing samples",
        ):
            # Load data
            kspace_full = test_data["kspace_full"][idx]
            kspace_undersampled = test_data["kspace_undersampled"][idx]
            mask = test_data["masks"][idx]
            phantom_gt = test_data["phantoms"][idx]  # Ground truth phantom

            # Ground truth image
            gt_image = phantom_gt

            # Zero-filled reconstruction (noisy input)
            zerofilled_image = kspace_to_image(kspace_undersampled)
            zerofilled_metrics = evaluate_reconstruction(gt_image, zerofilled_image)

            results["sample_idx"].append(idx)
            results["method"].append("Zero-filled")
            results["PSNR"].append(zerofilled_metrics["PSNR"])
            results["SSIM"].append(zerofilled_metrics["SSIM"])
            results["RMSE"].append(zerofilled_metrics["RMSE"])

            # Prepare tensors for models
            kspace_tensor = complex_to_tensor(kspace_undersampled).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)

            # K-space only U-Net reconstruction
            with torch.no_grad():
                pred_kspace = kspace_model(kspace_tensor, mask_tensor)
                pred_kspace_np = tensor_to_complex(pred_kspace[0].cpu())
                kspace_only_image = kspace_to_image(pred_kspace_np)

            kspace_only_metrics = evaluate_reconstruction(gt_image, kspace_only_image)

            results["sample_idx"].append(idx)
            results["method"].append("K-space U-Net")
            results["PSNR"].append(kspace_only_metrics["PSNR"])
            results["SSIM"].append(kspace_only_metrics["SSIM"])
            results["RMSE"].append(kspace_only_metrics["RMSE"])

            # Cascaded model reconstruction
            with torch.no_grad():
                reconstructed_kspace, reconstructed_image, refined_image = cascaded_model(kspace_tensor, mask_tensor)
                cascaded_image = refined_image.squeeze().cpu().numpy()

            cascaded_metrics = evaluate_reconstruction(gt_image, cascaded_image)

            results["sample_idx"].append(idx)
            results["method"].append("Cascaded Model")
            results["PSNR"].append(cascaded_metrics["PSNR"])
            results["SSIM"].append(cascaded_metrics["SSIM"])
            results["RMSE"].append(cascaded_metrics["RMSE"])

            # TV denoising applied to zero-filled reconstruction
            tv_image = tv_denoiser.denoise(zerofilled_image)
            tv_metrics = evaluate_reconstruction(gt_image, tv_image)

            results["sample_idx"].append(idx)
            results["method"].append("TV-Denoise")
            results["PSNR"].append(tv_metrics["PSNR"])
            results["SSIM"].append(tv_metrics["SSIM"])
            results["RMSE"].append(tv_metrics["RMSE"])

            # Visualize first few samples
            if idx < 5:
                _, axes = plt.subplots(3, 5, figsize=(20, 12))

                # Images - Row 1
                axes[0, 0].imshow(gt_image, cmap="gray")
                axes[0, 0].set_title("Ground Truth")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(zerofilled_image, cmap="gray")
                axes[0, 1].set_title(
                    f'Zero-filled\nPSNR: {zerofilled_metrics["PSNR"]:.2f}\nSSIM: {zerofilled_metrics["SSIM"]:.3f}'
                )
                axes[0, 1].axis("off")

                axes[0, 2].imshow(kspace_only_image, cmap="gray")
                axes[0, 2].set_title(
                    f'K-space U-Net\nPSNR: {kspace_only_metrics["PSNR"]:.2f}\nSSIM: {kspace_only_metrics["SSIM"]:.3f}'
                )
                axes[0, 2].axis("off")

                axes[0, 3].imshow(cascaded_image, cmap="gray")
                axes[0, 3].set_title(
                    f'Cascaded Model\nPSNR: {cascaded_metrics["PSNR"]:.2f}\nSSIM: {cascaded_metrics["SSIM"]:.3f}'
                )
                axes[0, 3].axis("off")

                axes[0, 4].imshow(tv_image, cmap="gray")
                axes[0, 4].set_title(
                    f'TV-Denoise (λ={lambda_tv})\nPSNR: {tv_metrics["PSNR"]:.2f}\nSSIM: {tv_metrics["SSIM"]:.3f}'
                )
                axes[0, 4].axis("off")

                # Error maps - Row 2
                axes[1, 0].imshow(mask, cmap="gray")
                axes[1, 0].set_title("Sampling Mask")
                axes[1, 0].axis("off")

                error_zero = np.abs(gt_image - zerofilled_image)
                im1 = axes[1, 1].imshow(error_zero, cmap="hot", vmin=0, vmax=0.3)
                axes[1, 1].set_title("Zero-filled Error")
                axes[1, 1].axis("off")
                plt.colorbar(im1, ax=axes[1, 1], fraction=0.046, pad=0.04)

                error_kspace = np.abs(gt_image - kspace_only_image)
                im2 = axes[1, 2].imshow(error_kspace, cmap="hot", vmin=0, vmax=0.3)
                axes[1, 2].set_title("K-space U-Net Error")
                axes[1, 2].axis("off")
                plt.colorbar(im2, ax=axes[1, 2], fraction=0.046, pad=0.04)

                error_cascaded = np.abs(gt_image - cascaded_image)
                im3 = axes[1, 3].imshow(error_cascaded, cmap="hot", vmin=0, vmax=0.3)
                axes[1, 3].set_title("Cascaded Model Error")
                axes[1, 3].axis("off")
                plt.colorbar(im3, ax=axes[1, 3], fraction=0.046, pad=0.04)

                error_tv = np.abs(gt_image - tv_image)
                im4 = axes[1, 4].imshow(error_tv, cmap="hot", vmin=0, vmax=0.3)
                axes[1, 4].set_title("TV-Denoise Error")
                axes[1, 4].axis("off")
                plt.colorbar(im4, ax=axes[1, 4], fraction=0.046, pad=0.04)

                # Difference maps - Row 3 (Cascaded vs others)
                axes[2, 0].axis('off')  # Empty

                diff_zero = cascaded_image - zerofilled_image
                im5 = axes[2, 1].imshow(diff_zero, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
                axes[2, 1].set_title("Cascaded - Zero-filled")
                axes[2, 1].axis("off")
                plt.colorbar(im5, ax=axes[2, 1], fraction=0.046, pad=0.04)

                diff_kspace = cascaded_image - kspace_only_image
                im6 = axes[2, 2].imshow(diff_kspace, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
                axes[2, 2].set_title("Cascaded - K-space U-Net")
                axes[2, 2].axis("off")
                plt.colorbar(im6, ax=axes[2, 2], fraction=0.046, pad=0.04)

                axes[2, 3].axis('off')  # Empty (self comparison)

                diff_tv = cascaded_image - tv_image
                im7 = axes[2, 4].imshow(diff_tv, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
                axes[2, 4].set_title("Cascaded - TV-Denoise")
                axes[2, 4].axis("off")
                plt.colorbar(im7, ax=axes[2, 4], fraction=0.046, pad=0.04)

                plt.suptitle(f"Sample {idx} - Cascaded Model Comparison", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"comparison_cascaded_sample_{idx}.png", dpi=300, bbox_inches='tight')
                plt.show()

    # Create results dataframe
    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n=== Summary Statistics ===")
    summary = results_df.groupby("method")[["PSNR", "SSIM", "RMSE"]].agg(
        ["mean", "std"]
    )
    print(summary)

    # Box plots
    _, axes = plt.subplots(1, 3, figsize=(18, 6))

    metrics = ["PSNR", "SSIM", "RMSE"]
    methods = ["Zero-filled", "K-space U-Net", "Cascaded Model", "TV-Denoise"]
    
    for i, metric in enumerate(metrics):
        data_to_plot = [
            results_df[results_df["method"] == method][metric].dropna()
            for method in methods
        ]

        bp = axes[i].boxplot(data_to_plot, labels=methods, patch_artist=True)
        
        # Color the boxes
        colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)

        axes[i].set_title(f"{metric} Distribution")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=15)

    plt.suptitle("Cascaded Model Performance Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("performance_comparison_cascaded.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save detailed results
    results_df.to_csv("reconstruction_results_cascaded.csv", index=False)
    print("\nDetailed results saved to 'reconstruction_results_cascaded.csv'")

    return results_df


def hyperparameter_search(
    dataset_path: str,
    num_samples: int = 5,
    lambda_values: np.ndarray = np.logspace(-2, 0, 10)
):
    """Search for optimal TV denoising parameter."""
    
    print("=== TV Denoising Hyperparameter Search ===")
    
    with h5py.File(dataset_path, "r") as f:
        test_data = f["test"]
        
        results = {
            'lambda': [],
            'avg_psnr': [],
            'avg_ssim': []
        }
        
        for lambda_tv in lambda_values:
            print(f"\nTesting λ = {lambda_tv:.4f}")
            tv_denoiser = TVDenoising(lambda_tv=lambda_tv, max_iter=70)
            
            psnr_values = []
            ssim_values = []
            
            for idx in range(min(num_samples, test_data["phantoms"].shape[0])):
                kspace_undersampled = test_data["kspace_undersampled"][idx]
                phantom_gt = test_data["phantoms"][idx]
                
                zerofilled_image = kspace_to_image(kspace_undersampled)
                
                # Apply TV denoising to zero-filled reconstruction
                tv_image = tv_denoiser.denoise(zerofilled_image)
                
                metrics = evaluate_reconstruction(phantom_gt, tv_image)
                psnr_values.append(metrics["PSNR"])
                ssim_values.append(metrics["SSIM"])
            
            avg_psnr = np.mean(psnr_values)
            avg_ssim = np.mean(ssim_values)
            
            results['lambda'].append(lambda_tv)
            results['avg_psnr'].append(avg_psnr)
            results['avg_ssim'].append(avg_ssim)
            
            print(f"  Average PSNR: {avg_psnr:.2f}")
            print(f"  Average SSIM: {avg_ssim:.3f}")
    
    # Plot results
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    ax1.semilogx(results['lambda'], results['avg_psnr'], 'o-')
    ax1.set_xlabel('λ (TV regularization)')
    ax1.set_ylabel('Average PSNR (dB)')
    ax1.set_title('PSNR vs TV Regularization')
    ax1.grid(True, alpha=0.3)
    
    ax2.semilogx(results['lambda'], results['avg_ssim'], 'o-')
    ax2.set_xlabel('λ (TV regularization)')
    ax2.set_ylabel('Average SSIM')
    ax2.set_title('SSIM vs TV Regularization')
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('tv_denoise_hyperparameter_search.png', dpi=300)
    plt.show()
    
    # Find optimal lambda
    optimal_idx = np.argmax(results['avg_psnr'])
    optimal_lambda = results['lambda'][optimal_idx]
    print(f"\nOptimal λ = {optimal_lambda:.4f} (based on PSNR)")
    
    return optimal_lambda


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "./mri_dataset.h5"
    CASCADED_MODEL_PATH = "./checkpoints_cascaded/last.ckpt"  # Path to your cascaded model checkpoint
    KSPACE_MODEL_PATH = "./adaptive-unet.pth"  # Path to your k-space model
    
    # First, find optimal lambda for TV denoising
    print("Finding optimal TV denoising parameters...")
    optimal_lambda = hyperparameter_search(
        dataset_path=DATASET_PATH,
        num_samples=5,
    )
    
    # Run comprehensive comparison
    print("\nRunning comprehensive comparison...")
    results = compare_methods(
        dataset_path=DATASET_PATH,
        cascaded_model_path=CASCADED_MODEL_PATH,
        kspace_model_path=KSPACE_MODEL_PATH,
        num_samples=20,
        lambda_tv=optimal_lambda
    )

    # Additional analysis
    print("\n=== Performance Improvement Analysis ===")
    
    # Calculate mean improvements
    methods = ["Zero-filled", "K-space U-Net", "Cascaded Model", "TV-Denoise"]
    mean_metrics = {}
    
    for method in methods:
        method_data = results[results["method"] == method]
        mean_metrics[method] = {
            "PSNR": method_data["PSNR"].mean(),
            "SSIM": method_data["SSIM"].mean(),
            "RMSE": method_data["RMSE"].mean()
        }
    
    zero_filled_psnr = mean_metrics["Zero-filled"]["PSNR"]
    
    print(f"Baseline (Zero-filled): {zero_filled_psnr:.2f} dB PSNR")
    print(f"K-space U-Net improvement: +{mean_metrics['K-space U-Net']['PSNR'] - zero_filled_psnr:.2f} dB")
    print(f"Cascaded Model improvement: +{mean_metrics['Cascaded Model']['PSNR'] - zero_filled_psnr:.2f} dB")
    print(f"TV-Denoise improvement: +{mean_metrics['TV-Denoise']['PSNR'] - zero_filled_psnr:.2f} dB")
    
    # Compare cascaded vs k-space only
    kspace_psnr = mean_metrics["K-space U-Net"]["PSNR"]
    cascaded_psnr = mean_metrics["Cascaded Model"]["PSNR"]
    print(f"\nCascaded vs K-space U-Net: +{cascaded_psnr - kspace_psnr:.2f} dB improvement")
    
    # Compare computational aspects
    print("\n=== Method Characteristics ===")
    print("Zero-filled: Instant, severe artifacts")
    print("K-space U-Net: Fast inference, requires training, removes some artifacts")
    print("Cascaded Model: Fast inference, requires training, two-stage refinement")
    print("TV-Denoise: Iterative optimization, removes streaking artifacts, preserves edges")
    
    print(f"\n✅ Comparison complete! Results saved to files.")