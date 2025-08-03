# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
from typing import Dict, Tuple, Union
import pandas as pd
from scipy.interpolate import interp1d
from scipy.ndimage import map_coordinates

def to_kspace(image: np.ndarray) -> np.ndarray:
    """Convert image to k-space using 2D FFT.
    
    Args:
        image: Input image array.
        
    Returns:
        K-space representation with DC component centered.
    """
    return fftshift(fft2(ifftshift(image)))

def from_kspace(kspace: np.ndarray) -> np.ndarray:
    """Reconstruct image from k-space using inverse 2D FFT.
    
    Args:
        kspace: K-space data with centered DC component.
        
    Returns:
        Reconstructed image (magnitude only).
    """
    return np.abs(fftshift(ifft2(ifftshift(kspace))))

def create_phantom(size: int = 256, phantom_type: str = 'shepp_logan') -> np.ndarray:
    """Create various test phantoms for MRI reconstruction testing.
    
    Args:
        size: Size of the phantom in pixels (creates size x size image).
        phantom_type: Type of phantom to create.
            - 'shepp_logan': Phantom with ellipses of varying brightness
            - 'resolution': Phantom with sinusoids of different frequencies
            for spatial resolution testing.

    Returns:
        2D phantom image as numpy array.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phantom = None
    
    if phantom_type == 'shepp_logan':
        # Simplified Shepp-Logan phantom
        phantom = np.zeros((size, size))
        
        # Large ellipse
        mask1 = ((X/0.69)**2 + (Y/0.92)**2) <= 1
        phantom[mask1] = 1.0
        
        # Smaller ellipses
        mask2 = ((X/0.3)**2 + ((Y-0.3)/0.35)**2) <= 1
        phantom[mask2] = 0.8
        
        mask3 = ((X/0.2)**2 + ((Y+0.3)/0.25)**2) <= 1
        phantom[mask3] = 0.6
        
    elif phantom_type == 'resolution':
        # Resolution test pattern
        phantom = np.zeros((size, size))
        # Stripes with different frequencies
        for i, freq in enumerate([2, 4, 8, 16, 32]):
            y_start = i * size // 5
            y_end = (i + 1) * size // 5
            x_pattern = np.sin(2 * np.pi * freq * x)
            phantom[y_start:y_end, :] = np.tile(x_pattern, (y_end - y_start, 1))
    
    return phantom

def create_undersampling_mask(
        shape: Tuple[int, int],
        pattern: str = 'random',
        acceleration: int = 4
    ) -> np.ndarray:
    """Create various undersampling patterns for accelerated MRI.
    
    Args:
        shape: Shape of the undersampling mask (height, width).
        pattern: Type of undersampling pattern to create.
            - 'random': Center 8% of k_x frequencies are fully sampled,
                with 1/acceleration of remaining lines sampled randomly.
            - 'regular': Every acceleration-th k_x line is sampled.
            - 'radial': 1/acceleration of available angular sections are
                sampled randomly for radial trajectory simulation.
        acceleration: Acceleration factor for undersampling.

    Returns:
        Binary mask where True indicates sampled k-space locations.
    """
    mask = np.zeros(shape, dtype=bool)
    
    if pattern == 'random':
        # Random lines with guaranteed center
        center_fraction = 0.08
        num_lines = shape[1]
        center_lines = int(center_fraction * num_lines)
        
        # Fully sample center
        center_start = (num_lines - center_lines) // 2
        mask[:, center_start:center_start + center_lines] = True
        
        # Randomly add more lines
        num_random = num_lines // acceleration - center_lines
        available_lines = [i for i in range(num_lines) 
                          if not mask[0, i]]
        random_lines = np.random.choice(available_lines, 
                                      size=num_random, 
                                      replace=False)
        mask[:, random_lines] = True
        
    elif pattern == 'regular':
        # Regular undersampling
        mask[:, ::acceleration] = True
        
    elif pattern == 'radial':
        # Radial undersampling (simulated)
        num_spokes = (shape[0] + shape[1]) // acceleration
        angles = np.linspace(0, np.pi, num_spokes, endpoint=False)
        center = np.array(shape) // 2
        
        for angle in angles:
            # Create line through k-space center
            for r in range(max(shape)):
                x = int(center[0] + r * np.cos(angle))
                y = int(center[1] + r * np.sin(angle))
                if 0 <= x < shape[0] and 0 <= y < shape[1]:
                    mask[x, y] = True
                x = int(center[0] - r * np.cos(angle))
                y = int(center[1] - r * np.sin(angle))
                if 0 <= x < shape[0] and 0 <= y < shape[1]:
                    mask[x, y] = True
    
    return mask

def linear_interpolation_kspace(kspace_undersampled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct image using linear interpolation in k_x-direction.
    
    Args:
        kspace_undersampled: Undersampled k-space data with missing values set to zero.
        mask: Binary mask indicating sampled k-space locations.
        
    Returns:
        Reconstructed image after k-space interpolation and inverse FFT.
    """
    kspace_filled = kspace_undersampled.copy()
    
    # For each row
    for i in range(kspace_filled.shape[0]):
        line = kspace_filled[i, :]
        mask_line = mask[i, :]
        
        if np.sum(mask_line) > 2:  # Enough points for interpolation
            # Find sampled positions
            sampled_indices = np.where(mask_line)[0]
            sampled_values = line[sampled_indices]
            
            # Interpolate real and imaginary parts separately
            real_interp = interpolate.interp1d(
                sampled_indices, np.real(sampled_values),
                kind='linear', fill_value='extrapolate'
            )
            imag_interp = interpolate.interp1d(
                sampled_indices, np.imag(sampled_values),
                kind='linear', fill_value='extrapolate'
            )
            
            # Fill missing values
            missing_indices = np.where(~mask_line)[0]
            kspace_filled[i, missing_indices] = (
                real_interp(missing_indices) + 
                1j * imag_interp(missing_indices)
            )
    
    return from_kspace(kspace_filled)

def radial_interpolation_kspace(k_space_undersampled: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Reconstruct image using radial interpolation in k-space.
    
    Args:
        k_space_undersampled: Complex undersampled k-space data.
        mask: Binary mask indicating sampled locations (True = sampled).
    
    Returns:
        Reconstructed image after radial k-space interpolation.
    """
    kspace_filled = k_space_undersampled.copy()
    
    # Get the center of k-space
    center_y, center_x = np.array(mask.shape) // 2
    
    # Create coordinate grids
    y, x = np.ogrid[:mask.shape[0], :mask.shape[1]]
    
    # Number of radial lines to use for interpolation
    num_angles = 180  # You can adjust this for better coverage
    
    for angle in np.linspace(0, np.pi, num_angles, endpoint=False):
        # Calculate the direction vector
        dx = np.cos(angle)
        dy = np.sin(angle)
        
        # Find the maximum radius for this angle
        max_radius = min(
            abs(center_x / dx) if dx != 0 else float('inf'),
            abs(center_y / dy) if dy != 0 else float('inf'),
            abs((mask.shape[1] - center_x - 1) / dx) if dx > 0 else float('inf'),
            abs((mask.shape[0] - center_y - 1) / dy) if dy > 0 else float('inf')
        )
        
        # Sample points along the radial line
        radii = np.linspace(-max_radius, max_radius, int(2 * max_radius))
        
        # Calculate coordinates along the line
        x_coords = center_x + radii * dx
        y_coords = center_y + radii * dy
        
        # Keep only valid coordinates
        valid_mask = (
            (x_coords >= 0) & (x_coords < mask.shape[1]) &
            (y_coords >= 0) & (y_coords < mask.shape[0])
        )
        
        x_coords = x_coords[valid_mask]
        y_coords = y_coords[valid_mask]
        radii = radii[valid_mask]
        
        # Get integer coordinates for mask checking
        x_int = np.round(x_coords).astype(int)
        y_int = np.round(y_coords).astype(int)
        
        # Find sampled points along this radial line
        sampled_indices = []
        sampled_values = []
        sampled_radii = []
        
        for i, (xi, yi, r) in enumerate(zip(x_int, y_int, radii)):
            if mask[yi, xi]:
                sampled_indices.append(i)
                # Use bilinear interpolation to get the exact value
                value = map_coordinates(
                    k_space_undersampled.real, 
                    [[y_coords[i]], [x_coords[i]]], 
                    order=1
                ) + 1j * map_coordinates(
                    k_space_undersampled.imag, 
                    [[y_coords[i]], [x_coords[i]]], 
                    order=1
                )
                sampled_values.append(value[0])
                sampled_radii.append(r)
        
        # Skip if not enough points for interpolation
        if len(sampled_values) < 2:
            continue
        
        # Create interpolation function for this radial line
        sampled_radii = np.array(sampled_radii)
        sampled_values = np.array(sampled_values)
        
        # Sort by radius to ensure monotonic interpolation
        sort_idx = np.argsort(sampled_radii)
        sampled_radii = sampled_radii[sort_idx]
        sampled_values = sampled_values[sort_idx]
        
        # Create interpolation functions for real and imaginary parts
        f_real = interp1d(
            sampled_radii, 
            sampled_values.real, 
            kind='cubic', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        f_imag = interp1d(
            sampled_radii, 
            sampled_values.imag, 
            kind='cubic', 
            bounds_error=False, 
            fill_value='extrapolate'
        )
        
        # Interpolate missing points along this radial line
        for i, (xi, yi, r) in enumerate(zip(x_int, y_int, radii)):
            if not mask[yi, xi]:
                # Only interpolate if within the range of sampled data
                if sampled_radii.min() <= r <= sampled_radii.max():
                    interpolated_value = f_real(r) + 1j * f_imag(r)
                    kspace_filled[yi, xi] = interpolated_value
    
    return from_kspace(kspace_filled)

def spline_interpolation_kspace(kspace_undersampled: np.ndarray, mask: np.ndarray, order: int = 3) -> np.ndarray:
    """Reconstruct image using spline interpolation in k-space.
    
    Args:
        kspace_undersampled: Undersampled k-space data.
        mask: Binary mask indicating sampled locations.
        order: Order of spline interpolation (default: 3 for cubic splines).
        
    Returns:
        Reconstructed image after spline interpolation in k-space.
    """
    kspace_filled = kspace_undersampled.copy()
    
    for i in range(kspace_filled.shape[0]):
        line = kspace_filled[i, :]
        mask_line = mask[i, :]
        
        if np.sum(mask_line) > order + 1:
            sampled_indices = np.where(mask_line)[0]
            sampled_values = line[sampled_indices]
            
            # Spline interpolation
            real_spline = interpolate.UnivariateSpline(
                sampled_indices, np.real(sampled_values),
                k=min(order, len(sampled_indices)-1), s=0
            )
            imag_spline = interpolate.UnivariateSpline(
                sampled_indices, np.imag(sampled_values),
                k=min(order, len(sampled_indices)-1), s=0
            )
            
            missing_indices = np.where(~mask_line)[0]
            kspace_filled[i, missing_indices] = (
                real_spline(missing_indices) + 
                1j * imag_spline(missing_indices)
            )
    
    return from_kspace(kspace_filled)

def low_pass_filter_recon(kspace_undersampled: np.ndarray, mask: np.ndarray, sigma: float = 1.0) -> np.ndarray:
    """Reconstruct image using zero-filling followed by Gaussian low-pass filtering.
    
    Args:
        kspace_undersampled: Undersampled k-space data.
        mask: Binary mask indicating sampled locations (unused but kept for consistency).
        sigma: Standard deviation for Gaussian filter (higher = more smoothing).
        
    Returns:
        Low-pass filtered reconstruction.
    """
    image = from_kspace(kspace_undersampled)
    
    # Gaussian low-pass filter
    filtered = gaussian_filter(image, sigma=sigma)
    
    return filtered

def analyze_frequency_response(phantom: np.ndarray, recons: Dict[str, np.ndarray]) -> None:
    """Analyze and visualize frequency response of different reconstruction methods.
    
    Args:
        phantom: Original phantom image.
        recons: Dictionary of reconstruction method names and their results.
    """
    _, axes = plt.subplots(2, len(recons) + 1, figsize=(20, 8))
    
    # Original k-space
    kspace_original = to_kspace(phantom)
    kspace_magnitude = np.log(np.abs(kspace_original) + 1e-8)
    
    axes[0, 0].imshow(kspace_magnitude, cmap='gray')
    axes[0, 0].set_title('Original k-space\n(log magnitude)')
    axes[0, 0].axis('off')
    
    # 1D profile through k-space center
    center_profile_original = np.abs(kspace_original[phantom.shape[0]//2, :])
    axes[1, 0].plot(center_profile_original)
    axes[1, 0].set_title('k-space Profile (center line)')
    axes[1, 0].set_yscale('log')
    axes[1, 0].set_xlabel('k-space position')
    axes[1, 0].set_ylabel('Magnitude')
    
    # Reconstructions
    for idx, (method_name, recon) in enumerate(recons.items(), 1):
        kspace_recon = to_kspace(recon)
        kspace_mag_recon = np.log(np.abs(kspace_recon) + 1e-8)
        
        axes[0, idx].imshow(kspace_mag_recon, cmap='gray')
        axes[0, idx].set_title(f'{method_name}\nk-space')
        axes[0, idx].axis('off')
        
        # Profile
        center_profile_recon = np.abs(kspace_recon[phantom.shape[0]//2, :])
        axes[1, idx].plot(center_profile_original, 'b-', alpha=0.5, label='Original')
        axes[1, idx].plot(center_profile_recon, 'r-', label='Recon')
        axes[1, idx].set_title(f'{method_name} Profile')
        axes[1, idx].set_yscale('log')
        axes[1, idx].set_xlabel('k-space position')
        axes[1, idx].legend()
    
    plt.tight_layout()
    plt.show()

def plot_reconstruction_comparison(
        phantom: np.ndarray, 
        mask: np.ndarray, 
        recons: Dict[str, np.ndarray], 
        title: str = "",
        acceleration: int = 4
    ) -> plt.Figure:
    """Compare different reconstruction methods with metrics visualization.
    
    Args:
        phantom: Original phantom image.
        mask: Undersampling mask used.
        recons: Dictionary of reconstruction method names and results.
        title: Title for the plot.
        acceleration: Acceleration factor used for undersampling.
        
    Returns:
        Matplotlib figure object.
    """
    n_methods = len(recons)
    fig, axes = plt.subplots(3, n_methods + 1, figsize=(20, 12))
    
    # Original and mask
    axes[0, 0].imshow(phantom, cmap='gray', vmin=0, vmax=1)
    axes[0, 0].set_title('Original')
    axes[0, 0].axis('off')
    
    axes[1, 0].imshow(mask, cmap='gray', aspect='auto')
    axes[1, 0].set_title(f'Sampling Pattern\n(Acceleration: {acceleration}x)')
    axes[1, 0].set_ylabel('k-space line')
    
    # Prepare metrics plot
    axes[2, 0].axis('off')
    
    # Reconstructions
    for idx, (method_name, recon) in enumerate(recons.items(), 1):
        # Reconstructed image
        axes[0, idx].imshow(recon, cmap='gray', vmin=0, vmax=1)
        axes[0, idx].set_title(method_name)
        axes[0, idx].axis('off')
        
        # Error map
        error = np.abs(phantom - recon)
        im = axes[1, idx].imshow(error, cmap='hot', vmin=0, vmax=0.5)
        axes[1, idx].set_title('Error Map')
        axes[1, idx].axis('off')
        
        # Calculate metrics
        psnr_val = psnr(phantom, recon, data_range=1.0)
        ssim_val = ssim(phantom, recon, data_range=1.0)
        rmse = np.sqrt(np.mean(error**2))
        
        # Display metrics
        metrics_str = f"{method_name}:\nPSNR: {psnr_val:.2f} dB\nSSIM: {ssim_val:.3f}\nRMSE: {rmse:.4f}"
        axes[2, idx].text(0.5, 0.5, metrics_str, 
                         ha='center', va='center', 
                         fontsize=10, 
                         bbox=dict(boxstyle="round,pad=0.3", 
                                  facecolor="lightgray"))
        axes[2, idx].axis('off')
    
    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    return fig

def create_summary_table(results: Dict[str, Dict]) -> pd.DataFrame:
    """Create summary table of reconstruction metrics for all methods and scenarios.
    
    Args:
        results: Dictionary containing reconstruction results for different scenarios.
        
    Returns:
        Pandas DataFrame with metrics summary.
    """
    summary_data = []
    
    for scenario, data in results.items():
        phantom = data['phantom']
        for method, recon in data['recons'].items():
            psnr_val = psnr(phantom, recon, data_range=1.0)
            ssim_val = ssim(phantom, recon, data_range=1.0)
            rmse = np.sqrt(np.mean((phantom - recon)**2))
            
            summary_data.append({
                'Scenario': scenario,
                'Method': method,
                'PSNR (dB)': f"{psnr_val:.2f}",
                'SSIM': f"{ssim_val:.3f}",
                'RMSE': f"{rmse:.4f}"
            })
    
    df = pd.DataFrame(summary_data)
    return df

# %%
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

size = 256
acceleration = 4
patterns = ['random', 'radial']
phantom_types = ['shepp_logan', 'resolution']

results = {}

for phantom_type in phantom_types:
    phantom = create_phantom(size, phantom_type)
    kspace_full = to_kspace(phantom)
    
    for pattern in patterns:
        # Create undersampling mask
        mask = create_undersampling_mask(kspace_full.shape, 
                                        pattern=pattern, 
                                        acceleration=acceleration)
        kspace_undersampled = kspace_full * mask
        
        # Various reconstructions
        recons = {
            'Inverse Fourier': from_kspace(kspace_undersampled),
            'Linear k-space': linear_interpolation_kspace(kspace_undersampled, mask),
            'Radial k-space': radial_interpolation_kspace(kspace_undersampled, mask),
            'Low-pass Filter': low_pass_filter_recon(kspace_undersampled, mask, sigma=2.0),
        }
        
        # Store results
        key = f"{phantom_type}_{pattern}"
        results[key] = {
            'phantom': phantom,
            'mask': mask,
            'recons': recons
        }

for key, data in results.items():
    fig = plot_reconstruction_comparison(
        data['phantom'], 
        data['mask'], 
        data['recons'],
        title=f"Reconstruction Comparison: {key}",
        acceleration=acceleration
    )
    plt.show()

# %% Analyze frequency response for Shepp-Logan with random sampling
data = results['shepp_logan_random']
analyze_frequency_response(data['phantom'], data['recons'])

# %%
# Show table
summary_df = create_summary_table(results)
print(summary_df.to_string(index=False))

# Best method per scenario
print("\n=== Best Method per Scenario (PSNR) ===")
for scenario in summary_df['Scenario'].unique():
    scenario_data = summary_df[summary_df['Scenario'] == scenario]
    # Convert PSNR back to float for comparison
    scenario_data['PSNR_val'] = scenario_data['PSNR (dB)'].astype(float)
    best = scenario_data.loc[scenario_data['PSNR_val'].idxmax()]
    print(f"{scenario}: {best['Method']} (PSNR: {best['PSNR (dB)']} dB)")
# %%

### Practical Tips:
# 1. **Sampling pattern matters!** Random sampling often works better than regular
# 2. **Center of k-space** should always be well sampled (low frequencies)
# 3. **Combine methods**: e.g., k-space interpolation first, then image domain filtering
# 4. **Iterative methods** (like POCS) can yield better results

### Next Steps:
# - Compressed Sensing with L1 regularization
# - Parallel imaging (SENSE/GRAPPA) when multi-coil data available
# - Deep neural networks for post-processing
# %%