# %%
import numpy as np
import matplotlib.pyplot as plt
from scipy import interpolate
from scipy.ndimage import gaussian_filter
from numpy.fft import fft2, ifft2, fftshift, ifftshift
from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr

def to_kspace(image):
    """Convert image to k-space."""
    return fftshift(fft2(ifftshift(image)))

def from_kspace(kspace):
    """Reconstruct image from k-space"""
    return np.abs(fftshift(ifft2(ifftshift(kspace))))

def create_phantom(size=256, phantom_type='shepp_logan'):
    """Create various test phantoms"""
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    phantom=None
    
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

def create_undersampling_mask(shape, pattern='random', acceleration=4):
    """Create various undersampling patterns"""
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
        num_spokes = shape[0] // acceleration
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


def linear_interpolation_kspace(kspace_undersampled, mask):
    """Linear interpolation in k-space"""
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


def spline_interpolation_kspace(kspace_undersampled, mask, order=3):
    """Spline interpolation in k-space"""
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

def low_pass_filter_recon(kspace_undersampled, mask, sigma=1.0):
    """Reconstruction with low-pass filter"""
    image = from_kspace(kspace_undersampled)
    
    # Gaussian low-pass filter
    filtered = gaussian_filter(image, sigma=sigma)
    
    return filtered


def analyze_frequency_response(phantom, recons):
    """Analyze frequency response of different methods"""
    _, axes = plt.subplots(2, len(recons) + 1, figsize=(20, 8))
    
    # Original k-space
    kspace_original = to_kspace(phantom)
    kspace_magnitude = np.log(np.abs(kspace_original) + 1e-8)
    
    axes[0, 0].imshow(kspace_magnitude, cmap='gray')
    axes[0, 0].set_title('Original k-space\n(log magnitude)')
    axes[0, 0].axis('off')
    
    # 1D profile through k-space center
    center_profile_original = np.abs(kspace_original[size//2, :])
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
        center_profile_recon = np.abs(kspace_recon[size//2, :])
        axes[1, idx].plot(center_profile_original, 'b-', alpha=0.5, label='Original')
        axes[1, idx].plot(center_profile_recon, 'r-', label='Recon')
        axes[1, idx].set_title(f'{method_name} Profile')
        axes[1, idx].set_yscale('log')
        axes[1, idx].set_xlabel('k-space position')
        axes[1, idx].legend()
    
    plt.tight_layout()
    plt.show()


def plot_reconstruction_comparison(phantom, mask, recons, title=""):
    """Compare different reconstruction methods"""
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


# %%
plt.rcParams['figure.figsize'] = (15, 10)
plt.rcParams['font.size'] = 12

size = 256
acceleration = 4
patterns = ['random', 'regular']
phantom_types = ['shepp_logan', 'resolution']

results = {}
# %%
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
            'Spline k-space': spline_interpolation_kspace(kspace_undersampled, mask),
            'Low-pass Filter': low_pass_filter_recon(kspace_undersampled, mask, sigma=2.0),
        }
        
        # Store results
        key = f"{phantom_type}_{pattern}"
        results[key] = {
            'phantom': phantom,
            'mask': mask,
            'recons': recons
        }

# %%
for key, data in results.items():
    fig = plot_reconstruction_comparison(
        data['phantom'], 
        data['mask'], 
        data['recons'],
        title=f"Reconstruction Comparison: {key}"
    )
    plt.show()


# Analyze frequency response for Shepp-Logan with random sampling
data = results['shepp_logan_random']
analyze_frequency_response(data['phantom'], data['recons'])