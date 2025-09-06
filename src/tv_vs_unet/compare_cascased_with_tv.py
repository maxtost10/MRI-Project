import torch
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

# Import reusable functions from existing script
from compare_methods import (
    complex_to_tensor, tensor_to_complex, kspace_to_image, 
    evaluate_reconstruction, TVDenoising, hyperparameter_search
)

# Import models
from train_unet import AdaptiveUNet
from train_cascaded_refinement import CascadedMRIReconstruction


def compare_cascaded_methods(
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

    # Load standalone k-space model
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

    # Results storage
    results = {"sample_idx": [], "method": [], "PSNR": [], "SSIM": [], "RMSE": []}

    # Load test data
    import h5py
    with h5py.File(dataset_path, "r") as f:
        test_data = f["test"]

        for idx in tqdm(range(min(num_samples, test_data["phantoms"].shape[0])), 
                       desc="Processing samples"):
            
            # Load data
            kspace_full = test_data["kspace_full"][idx]
            kspace_undersampled = test_data["kspace_undersampled"][idx]
            mask = test_data["masks"][idx]
            phantom_gt = test_data["phantoms"][idx]  # Ground truth phantom

            # Prepare tensors
            kspace_tensor = complex_to_tensor(kspace_undersampled).unsqueeze(0).to(device)
            mask_tensor = torch.from_numpy(mask.astype(np.float32)).unsqueeze(0).to(device)

            # Zero-filled reconstruction
            zerofilled_image = kspace_to_image(kspace_undersampled)
            zerofilled_metrics = evaluate_reconstruction(phantom_gt, zerofilled_image)
            
            results["sample_idx"].append(idx)
            results["method"].append("Zero-filled")
            for key, val in zerofilled_metrics.items():
                results[key].append(val)

            # K-space only U-Net
            with torch.no_grad():
                pred_kspace = kspace_model(kspace_tensor, mask_tensor)
                pred_kspace_np = tensor_to_complex(pred_kspace[0].cpu())
                kspace_only_image = kspace_to_image(pred_kspace_np)

            kspace_only_metrics = evaluate_reconstruction(phantom_gt, kspace_only_image)
            
            results["sample_idx"].append(idx)
            results["method"].append("K-space U-Net")
            for key, val in kspace_only_metrics.items():
                results[key].append(val)

            # Cascaded model
            with torch.no_grad():
                _, _, refined_image = cascaded_model(kspace_tensor, mask_tensor)
                cascaded_image = refined_image.squeeze().cpu().numpy()

            cascaded_metrics = evaluate_reconstruction(phantom_gt, cascaded_image)
            
            results["sample_idx"].append(idx)
            results["method"].append("Cascaded Model")
            for key, val in cascaded_metrics.items():
                results[key].append(val)

            # TV denoising
            tv_image = tv_denoiser.denoise(zerofilled_image)
            tv_metrics = evaluate_reconstruction(phantom_gt, tv_image)
            
            results["sample_idx"].append(idx)
            results["method"].append("TV-Denoise")
            for key, val in tv_metrics.items():
                results[key].append(val)

            # Visualize first few samples
            if idx < 3:
                fig, axes = plt.subplots(3, 5, figsize=(20, 12))

                # Row 1: Images
                axes[0, 0].imshow(phantom_gt, cmap="gray")
                axes[0, 0].set_title("Ground Truth")
                axes[0, 0].axis("off")

                axes[0, 1].imshow(zerofilled_image, cmap="gray")
                axes[0, 1].set_title(f'Zero-filled\nPSNR: {zerofilled_metrics["PSNR"]:.1f}')
                axes[0, 1].axis("off")

                axes[0, 2].imshow(kspace_only_image, cmap="gray")
                axes[0, 2].set_title(f'K-space U-Net\nPSNR: {kspace_only_metrics["PSNR"]:.1f}')
                axes[0, 2].axis("off")

                axes[0, 3].imshow(cascaded_image, cmap="gray")
                axes[0, 3].set_title(f'Cascaded Model\nPSNR: {cascaded_metrics["PSNR"]:.1f}')
                axes[0, 3].axis("off")

                axes[0, 4].imshow(tv_image, cmap="gray")
                axes[0, 4].set_title(f'TV-Denoise\nPSNR: {tv_metrics["PSNR"]:.1f}')
                axes[0, 4].axis("off")

                # Row 2: Error maps
                axes[1, 0].imshow(mask, cmap="gray")
                axes[1, 0].set_title("Sampling Mask")
                axes[1, 0].axis("off")

                for i, (img, title) in enumerate([
                    (zerofilled_image, "Zero-filled Error"),
                    (kspace_only_image, "K-space U-Net Error"), 
                    (cascaded_image, "Cascaded Error"),
                    (tv_image, "TV-Denoise Error")
                ], 1):
                    error = np.abs(phantom_gt - img)
                    im = axes[1, i].imshow(error, cmap="hot", vmin=0, vmax=0.3)
                    axes[1, i].set_title(title)
                    axes[1, i].axis("off")
                    plt.colorbar(im, ax=axes[1, i], fraction=0.046, pad=0.04)

                # Row 3: Cascaded vs others difference maps
                axes[2, 0].axis('off')
                
                for i, (img, title) in enumerate([
                    (zerofilled_image, "Cascaded - Zero-filled"),
                    (kspace_only_image, "Cascaded - K-space U-Net"),
                    (cascaded_image, "Self (empty)"),
                    (tv_image, "Cascaded - TV-Denoise")
                ], 1):
                    if i == 3:  # Self comparison - empty
                        axes[2, i].axis('off')
                        continue
                    diff = cascaded_image - img
                    im = axes[2, i].imshow(diff, cmap="RdBu_r", vmin=-0.2, vmax=0.2)
                    axes[2, i].set_title(title)
                    axes[2, i].axis("off")
                    plt.colorbar(im, ax=axes[2, i], fraction=0.046, pad=0.04)

                plt.suptitle(f"Sample {idx} - Cascaded Model Comparison", fontsize=16)
                plt.tight_layout()
                plt.savefig(f"plots_tv_cascaded_comparison/comparison_cascaded_sample_{idx}.png", dpi=300, bbox_inches='tight')
                plt.show()

    # Convert to DataFrame and analyze
    import pandas as pd
    results_df = pd.DataFrame(results)

    # Summary statistics
    print("\n=== Summary Statistics ===")
    summary = results_df.groupby("method")[["PSNR", "SSIM", "RMSE"]].agg(["mean", "std"])
    print(summary)

    # Box plots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    methods = ["Zero-filled", "K-space U-Net", "Cascaded Model", "TV-Denoise"]
    colors = ['lightgray', 'lightblue', 'lightgreen', 'lightcoral']
    
    for i, metric in enumerate(["PSNR", "SSIM", "RMSE"]):
        data_to_plot = [results_df[results_df["method"] == method][metric].dropna() 
                       for method in methods]
        bp = axes[i].boxplot(data_to_plot, labels=methods, patch_artist=True)
        
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            
        axes[i].set_title(f"{metric} Distribution")
        axes[i].set_ylabel(metric)
        axes[i].grid(True, alpha=0.3)
        axes[i].tick_params(axis='x', rotation=15)

    plt.suptitle("Cascaded Model Performance Comparison", fontsize=16)
    plt.tight_layout()
    plt.savefig("plots_tv_cascaded_comparison/performance_comparison_cascaded.png", dpi=300, bbox_inches='tight')
    plt.show()

    # Save results
    results_df.to_csv("reconstruction_results_cascaded.csv", index=False)
    print("\nDetailed results saved to 'reconstruction_results_cascaded.csv'")

    return results_df


if __name__ == "__main__":
    # Configuration
    DATASET_PATH = "./mri_dataset.h5"
    CASCADED_MODEL_PATH = "./checkpoints_cascaded/last.ckpt"
    KSPACE_MODEL_PATH = "./adaptive-unet.pth"
    
    # Find optimal TV parameters (reusing existing function)
    print("Finding optimal TV denoising parameters...")
    optimal_lambda = hyperparameter_search(
        dataset_path=DATASET_PATH,
        num_samples=5,
    )
    
    # Run cascaded comparison
    print("\nRunning cascaded model comparison...")
    results = compare_cascaded_methods(
        dataset_path=DATASET_PATH,
        cascaded_model_path=CASCADED_MODEL_PATH,
        kspace_model_path=KSPACE_MODEL_PATH,
        num_samples=20,
        lambda_tv=optimal_lambda
    )

    # Performance analysis
    print("\n=== Performance Improvement Analysis ===")
    methods = ["Zero-filled", "K-space U-Net", "Cascaded Model", "TV-Denoise"]
    mean_psnr = {method: results[results["method"] == method]["PSNR"].mean() 
                 for method in methods}
    
    baseline = mean_psnr["Zero-filled"]
    print(f"Baseline (Zero-filled): {baseline:.2f} dB PSNR")
    for method in methods[1:]:
        improvement = mean_psnr[method] - baseline
        print(f"{method} improvement: +{improvement:.2f} dB")
    
    # Cascaded vs K-space comparison
    cascaded_vs_kspace = mean_psnr["Cascaded Model"] - mean_psnr["K-space U-Net"]
    print(f"\nCascaded vs K-space U-Net: +{cascaded_vs_kspace:.2f} dB improvement")
    
    print("\n=== Method Characteristics ===")
    print("Zero-filled: Instant, severe artifacts")
    print("K-space U-Net: Fast inference, removes some artifacts")
    print("Cascaded Model: Two-stage refinement, best quality")
    print("TV-Denoise: Iterative optimization, edge-preserving")
    
    print(f"\n✅ Cascaded comparison complete!")