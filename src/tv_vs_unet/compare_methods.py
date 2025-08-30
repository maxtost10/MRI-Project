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

# Import from your existing script
from train_unet import AdaptiveUNet


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
    model_path: str,
    num_samples: int = 20,
    lambda_tv: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Compare U-Net and TV denoising reconstruction methods."""

    # Load model
    print("Loading model...")
    checkpoint = torch.load(model_path, map_location=device, weights_only=False)
    hparams = checkpoint["hparams"]

    model = AdaptiveUNet(
        in_channels=hparams.get("in_channels", 2),
        out_channels=hparams.get("out_channels", 2),
        acceleration=hparams.get("acceleration", 4),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.to(device)
    model.eval()

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

            # Ground truth reconstruction
            gt_image = kspace_to_image(kspace_full)

            # Zero-filled reconstruction (noisy input)
            zerofilled_image = kspace_to_image(kspace_undersampled)
            zerofilled_metrics = evaluate_reconstruction(gt_image, zerofilled_image)

            results["sample_idx"].append(idx)
            results["method"].append("Zero-filled")
            results["PSNR"].append(zerofilled_metrics["PSNR"])
            results["SSIM"].append(zerofilled_metrics["SSIM"])
            results["RMSE"].append(zerofilled_metrics["RMSE"])

            # U-Net reconstruction
            with torch.no_grad():
                kspace_tensor = (
                    complex_to_tensor(kspace_undersampled).unsqueeze(0).to(device)
                )
                mask_tensor = (
                    torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)
                )

                pred_kspace = model(kspace_tensor, mask_tensor)
                pred_kspace_np = tensor_to_complex(pred_kspace[0].cpu())
                unet_image = kspace_to_image(pred_kspace_np)

            unet_metrics = evaluate_reconstruction(gt_image, unet_image)

            results["sample_idx"].append(idx)
            results["method"].append("U-Net")
            results["PSNR"].append(unet_metrics["PSNR"])
            results["SSIM"].append(unet_metrics["SSIM"])
            results["RMSE"].append(unet_metrics["RMSE"])

            # TV denoising (simple artifact removal)
            print(f"\nSample {idx}: Running TV denoising...")
            tv_image = tv_denoiser.denoise(zerofilled_image)
            tv_metrics = evaluate_reconstruction(gt_image, tv_image)

            results["sample_idx"].append(idx)
            results["method"].append("TV-Denoise")
            results["PSNR"].append(tv_metrics["PSNR"])
            results["SSIM"].append(tv_metrics["SSIM"])
            results["RMSE"].append(tv_metrics["RMSE"])

            # Visualize first few samples
            if idx < 5:
                _, axes = plt.subplots(2, 4, figsize=(16, 8))

                # Images
                axes[0, 0].imshow(gt_image, cmap="gray")
                axes[0, 0].set_title("Ground Truth")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(zerofilled_image, cmap="gray")
                axes[0, 1].set_title(
                    f'Zero-filled\nPSNR: {zerofilled_metrics["PSNR"]:.2f}\nSSIM: {zerofilled_metrics["SSIM"]:.2f}'
                )
                axes[0, 1].axis("off")

                axes[0, 2].imshow(unet_image, cmap="gray")
                axes[0, 2].set_title(
                    f'U-Net\nPSNR: {unet_metrics["PSNR"]:.2f}\nSSIM: {unet_metrics["SSIM"]:.2f}'
                )
                axes[0, 2].axis("off")

                axes[0, 3].imshow(tv_image, cmap="gray")
                axes[0, 3].set_title(
                    f'TV (λ={lambda_tv})\nPSNR: {tv_metrics["PSNR"]:.2f}\nSSIM: {tv_metrics["SSIM"]:.2f}'
                )
                axes[0, 3].axis("off")

                # Error maps
                axes[1, 0].imshow(mask, cmap="gray")
                axes[1, 0].set_title("Sampling Mask")
                axes[1, 0].axis("off")

                error_zero = np.abs(gt_image - zerofilled_image)
                axes[1, 1].imshow(error_zero, cmap="hot", vmin=0, vmax=0.5)
                axes[1, 1].set_title("Zero-filled Error")
                axes[1, 1].axis("off")

                error_unet = np.abs(gt_image - unet_image)
                axes[1, 2].imshow(error_unet, cmap="hot", vmin=0, vmax=0.5)
                axes[1, 2].set_title("U-Net Error")
                axes[1, 2].axis("off")

                error_tv = np.abs(gt_image - tv_image)
                axes[1, 3].imshow(error_tv, cmap="hot", vmin=0, vmax=0.5)
                axes[1, 3].set_title("TV-Denoise Error")
                axes[1, 3].axis("off")

                plt.suptitle(f"Sample {idx} Comparison")
                plt.tight_layout()
                plt.savefig(f"comparison_tv_denoise_sample_{idx}.png", dpi=150)
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
    _, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["PSNR", "SSIM", "RMSE"]
    for i, metric in enumerate(metrics):
        data_to_plot = [
            results_df[results_df["method"] == method][metric].dropna()
            for method in ["Zero-filled", "U-Net", "TV-Denoise"]
        ]

        axes[i].boxplot(data_to_plot, labels=["Zero-filled", "U-Net", "TV-Denoise"])
        axes[i].set_title(f"{metric} Distribution")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Reconstruction Performance Comparison")
    plt.tight_layout()
    plt.savefig("performance_comparison_tv_denoise.png", dpi=150)
    plt.show()

    # Save detailed results
    results_df.to_csv("reconstruction_results_tv_denoise.csv", index=False)
    print("\nDetailed results saved to 'reconstruction_results_tv_denoise.csv'")

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
            print(f"\nTesting λ = {lambda_tv}")
            tv_denoiser = TVDenoising(lambda_tv=lambda_tv, max_iter=70)
            
            psnr_values = []
            ssim_values = []
            
            for idx in range(min(num_samples, test_data["phantoms"].shape[0])):
                kspace_full = test_data["kspace_full"][idx]
                kspace_undersampled = test_data["kspace_undersampled"][idx]
                
                gt_image = kspace_to_image(kspace_full)
                zerofilled_image = kspace_to_image(kspace_undersampled)
                
                # Apply TV denoising to zero-filled reconstruction
                tv_image = tv_denoiser.denoise(zerofilled_image)
                
                metrics = evaluate_reconstruction(gt_image, tv_image)
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
    plt.savefig('tv_denoise_hyperparameter_search.png', dpi=150)
    plt.show()
    
    # Find optimal lambda
    optimal_idx = np.argmax(results['avg_psnr'])
    optimal_lambda = results['lambda'][optimal_idx]
    print(f"\nOptimal λ = {optimal_lambda} (based on PSNR)")
    
    return optimal_lambda


if __name__ == "__main__":
    # First, find optimal lambda for denoising
    optimal_lambda = hyperparameter_search(
        dataset_path="./mri_dataset.h5",
        num_samples=5,
    )
    
    # Run comparison with optimal lambda
    results = compare_methods(
        dataset_path="./mri_dataset.h5",
        model_path="./adaptive-unet.pth",
        num_samples=20,
        lambda_tv=optimal_lambda
    )

    # Additional analysis
    print("\n=== Performance Improvement over Zero-filled ===")
    zero_filled_psnr = results[results["method"] == "Zero-filled"]["PSNR"].mean()
    unet_psnr = results[results["method"] == "U-Net"]["PSNR"].mean()
    tv_psnr = results[results["method"] == "TV-Denoise"]["PSNR"].mean()

    print(f"U-Net improvement: {unet_psnr - zero_filled_psnr:.2f} dB")
    print(f"TV-Denoise improvement: {tv_psnr - zero_filled_psnr:.2f} dB")
    
    # Compare computational aspects
    print("\n=== Method Characteristics ===")
    print("U-Net: Fast inference, requires training data, may hallucinate")
    print("TV-Denoise: Simple post-processing, removes streaking artifacts, preserves edges")