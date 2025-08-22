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


class TVMinimization:
    """Total Variation minimization for MRI reconstruction using FISTA algorithm."""
    
    def __init__(self, lambda_tv: float = 0.01, max_iter: int = 100, tol: float = 1e-4):
        """
        Initialize TV minimization solver.
        
        Args:
            lambda_tv: Regularization parameter for TV penalty
            max_iter: Maximum number of iterations
            tol: Convergence tolerance
        """
        self.lambda_tv = lambda_tv
        self.max_iter = max_iter
        self.tol = tol
        
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
    
    def _prox_tv(self, x: np.ndarray, lambda_param: float, niter: int = 20) -> np.ndarray:
        """
        Proximal operator for TV regularization using Chambolle's algorithm.
        
        This solves: argmin_u { 0.5 * ||u - x||^2 + lambda_param * TV(u) }
        """
        # Initialize dual variables
        px = np.zeros_like(x)
        py = np.zeros_like(x)
        
        tau = 0.25  # Step size for dual problem
        
        for _ in range(niter):
            # Compute divergence
            div_p = self._compute_divergence(px, py)
            
            # Update primal variable: u = x - λ·div(p)
            u_tilde = x - lambda_param * div_p
            
            # Compute gradient of u_tilde
            dx, dy = self._compute_gradient(u_tilde)
            
            # Update dual variables with projection
            px_new = px + tau * dx
            py_new = py + tau * dy
            
            # Project onto unit ball (element-wise)
            norm_p = np.sqrt(px_new**2 + py_new**2)
            norm_p = np.maximum(norm_p, 1.0)
            
            # p[i] = sign(∇u[i]) to maximize λ⟨p, ∇u⟩
            px = px_new / norm_p 
            py = py_new / norm_p
        
        # Final update
        return x - lambda_param * self._compute_divergence(px, py)
    
    def _data_consistency(self, x: np.ndarray, y: np.ndarray, mask: np.ndarray, 
                         step_size: float = 1.0) -> np.ndarray:
        """
        Gradient step for data consistency term.
        
        Computes: x - step_size * A^H(Ax - y)
        where A is the undersampled Fourier operator.
        """
        # Forward: image to k-space
        x_kspace = fftshift(fft2(ifftshift(x)))
        
        # Compute residual in k-space (only at sampled locations)
        residual = np.zeros_like(x_kspace)
        residual[mask] = x_kspace[mask] - y[mask]
        
        # Backward: k-space to image
        grad = fftshift(ifft2(ifftshift(residual)))
        
        # Apply gradient step (real part for magnitude image)
        return x - step_size * np.real(grad)
    
    def reconstruct(self, kspace_undersampled: np.ndarray, mask: np.ndarray) -> np.ndarray:
        """
        Reconstruct image from undersampled k-space using TV minimization.
        
        Uses FISTA (Fast Iterative Shrinkage-Thresholding Algorithm).
        """
        # Initialize with zero-filled reconstruction
        x = kspace_to_image(kspace_undersampled)
        
        # FISTA parameters
        t = 1.0
        x_old = x.copy()
        
        # Lipschitz constant (approximate)
        L = 2.0  # For normalized FFT
        step_size = 1.0 / L
        
        # Main FISTA loop
        pbar = tqdm(range(self.max_iter), desc="TV minimization", leave=False)
        
        for iteration in pbar:
            # Momentum term (FISTA acceleration)
            if iteration > 0:
                t_new = (1 + np.sqrt(1 + 4 * t**2)) / 2
                y = x + ((t - 1) / t_new) * (x - x_old)
                t = t_new
            else:
                y = x
            
            # Data consistency gradient step
            x_temp = self._data_consistency(y, kspace_undersampled, mask, step_size)
            
            # Proximal TV step
            x_new = self._prox_tv(x_temp, self.lambda_tv * step_size)
            
            # Check convergence
            rel_change = np.linalg.norm(x_new - x) / (np.linalg.norm(x) + 1e-8)
            pbar.set_postfix({'rel_change': f'{rel_change:.2e}'})
            
            if rel_change < self.tol:
                pbar.close()
                break
            
            # Update
            x_old = x
            x = x_new
        
        return x


def evaluate_reconstruction(
    ground_truth: np.ndarray, reconstruction: np.ndarray
) -> Dict[str, float]:
    """Calculate reconstruction metrics."""
    # Ensure same scale
    gt_normalized = ground_truth / np.max(ground_truth)
    recon_normalized = reconstruction / np.max(reconstruction)

    metrics = {
        "PSNR": psnr(gt_normalized, recon_normalized, data_range=1.0),
        "SSIM": ssim(gt_normalized, recon_normalized, data_range=1.0),
        "RMSE": np.sqrt(np.mean((gt_normalized - recon_normalized) ** 2)),
    }

    return metrics


def compare_methods(
    dataset_path: str,
    model_path: str,
    num_samples: int = 20,
    lambda_tv: float = 0.01,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
):
    """Compare U-Net and TV minimization reconstruction methods."""

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

    # Initialize TV solver
    tv_solver = TVMinimization(lambda_tv=lambda_tv, max_iter=100)

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

            # Zero-filled reconstruction
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

            # TV minimization reconstruction
            print(f"\nSample {idx}: Running TV minimization...")
            tv_image = tv_solver.reconstruct(kspace_undersampled, mask)
            tv_metrics = evaluate_reconstruction(gt_image, tv_image)

            results["sample_idx"].append(idx)
            results["method"].append("TV")
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
                axes[1, 3].set_title("TV Error")
                axes[1, 3].axis("off")

                plt.suptitle(f"Sample {idx} Comparison")
                plt.tight_layout()
                plt.savefig(f"comparison_tv_sample_{idx}.png", dpi=150)
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
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    metrics = ["PSNR", "SSIM", "RMSE"]
    for i, metric in enumerate(metrics):
        data_to_plot = [
            results_df[results_df["method"] == method][metric].dropna()
            for method in ["Zero-filled", "U-Net", "TV"]
        ]

        axes[i].boxplot(data_to_plot, labels=["Zero-filled", "U-Net", "TV"])
        axes[i].set_title(f"{metric} Distribution")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)

    plt.suptitle("Reconstruction Performance Comparison")
    plt.tight_layout()
    plt.savefig("performance_comparison_tv.png", dpi=150)
    plt.show()

    # Save detailed results
    results_df.to_csv("reconstruction_results_tv.csv", index=False)
    print("\nDetailed results saved to 'reconstruction_results_tv.csv'")

    return results_df


def hyperparameter_search(
    dataset_path: str,
    num_samples: int = 5,
    lambda_values: list = [0.001, 0.005, 0.01, 0.05, 0.1]
):
    """Search for optimal TV regularization parameter."""
    
    print("=== TV Hyperparameter Search ===")
    
    with h5py.File(dataset_path, "r") as f:
        test_data = f["train"]
        
        results = {
            'lambda': [],
            'avg_psnr': [],
            'avg_ssim': []
        }
        
        for lambda_tv in lambda_values:
            print(f"\nTesting λ = {lambda_tv}")
            tv_solver = TVMinimization(lambda_tv=lambda_tv, max_iter=100)
            
            psnr_values = []
            ssim_values = []
            
            for idx in range(min(num_samples, test_data["phantoms"].shape[0])):
                kspace_full = test_data["kspace_full"][idx]
                kspace_undersampled = test_data["kspace_undersampled"][idx]
                mask = test_data["masks"][idx]
                
                gt_image = kspace_to_image(kspace_full)
                tv_image = tv_solver.reconstruct(kspace_undersampled, mask)
                
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
    plt.savefig('tv_hyperparameter_search.png', dpi=150)
    plt.show()
    
    # Find optimal lambda
    optimal_idx = np.argmax(results['avg_psnr'])
    optimal_lambda = results['lambda'][optimal_idx]
    print(f"\nOptimal λ = {optimal_lambda} (based on PSNR)")
    
    return optimal_lambda


# %%
if __name__ == "__main__":
    # First, find optimal lambda
    optimal_lambda = hyperparameter_search(
        dataset_path="./mri_dataset.h5",
        num_samples=5,
        lambda_values=[0.001, 0.005, 0.01, 0.05, 0.1]
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
    tv_psnr = results[results["method"] == "TV"]["PSNR"].mean()

    print(f"U-Net improvement: {unet_psnr - zero_filled_psnr:.2f} dB")
    print(f"TV improvement: {tv_psnr - zero_filled_psnr:.2f} dB")
    
    # Compare computational aspects
    print("\n=== Method Characteristics ===")
    print("U-Net: Fast inference, requires training data, may hallucinate")
    print("TV: No training needed, slower iterative optimization, preserves edges")

# %%