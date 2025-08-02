import numpy as np
from numpy.fft import fft2, ifft2, fftshift, ifftshift
import matplotlib.pyplot as plt

def create_shepp_logan_phantom(size: int=256):
    """Create a Test-Phantom.
    
    Args:
        - size (int): The size of the phantom in pixels.
    """
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    # Circle
    phantom = np.zeros((size, size))
    mask = X**2 + Y**2 <= 0.5**2
    phantom[mask] = 1.0
    
    # Second smaller circle
    mask2 = X**2 + Y**2 <= 0.2**2
    phantom[mask2] = 0.5
    
    return phantom

def to_kspace(image):
    """Convert image to k-space"""
    return fftshift(fft2(ifftshift(image)))

def from_kspace(kspace):
    """Reconstruct image from k-space"""
    return np.abs(fftshift(ifft2(ifftshift(kspace))))

def create_random_mask(shape, acceleration=4):
    """Creates a random subsampling mask."""
    mask = np.zeros(shape)
    # Keep central k-space lines (low frequencies)
    center_fraction = 0.08
    num_lines = shape[1]
    center_lines = int(center_fraction * num_lines)
    
    # Completely sample center
    center_start = (num_lines - center_lines) // 2
    mask[:, center_start:center_start + center_lines] = 1
    
    # Randomly add further lines that will be kept
    num_random_lines = int((num_lines - center_lines) // acceleration)
    # np.random.choice(list, n) chooses randomly n elements from the list
    random_lines = np.random.choice(
        [i for i in range(num_lines) if mask[0, i] == 0],
        size=num_random_lines,
        replace=False
    )
    mask[:, random_lines] = 1
    
    return mask


# %% Create Synthetic Data
phantom = create_shepp_logan_phantom()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(phantom, cmap='gray')
plt.title('Original Phantom')
plt.colorbar()

# %% Transform to k-space
kspace = to_kspace(phantom)

plt.figure(figsize=(10, 5))
plt.subplot(122)
plt.imshow(np.log(np.abs(kspace) + 1e-9), cmap='gray')
plt.title('K-Space (log magnitude)')
plt.colorbar()
plt.show()

# %% Apply random mask to higher frequencies

mask = create_random_mask(kspace.shape, acceleration=3)
undersampled_kspace = kspace * mask
zero_filled_recon = from_kspace(undersampled_kspace)

# %% Analyse reconstruction

mse = np.mean((phantom - zero_filled_recon)**2)
psnr = 10 * np.log10(1.0 / mse)
print(f'PSNR: {psnr:.2f} dB')

fig, axes = plt.subplots(2, 2, figsize=(15, 10))

axes[0, 0].imshow(phantom, cmap='gray')
axes[0, 0].set_title('Original')

axes[0, 1].imshow(mask, cmap='gray', aspect='auto')
axes[0, 1].set_title('Sampling Pattern')

axes[1, 0].imshow(zero_filled_recon, cmap='gray')
axes[1, 0].set_title('Zero-Filled Reconstruction')

axes[1, 1].imshow(np.abs(phantom - zero_filled_recon), cmap='hot')
axes[1, 1].set_title('Error Map')

plt.tight_layout()
plt.show()