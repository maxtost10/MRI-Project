# %%
import numpy as np
import h5py
from pathlib import Path
from typing import Tuple, List
import matplotlib.pyplot as plt
from tqdm import tqdm

# Import functions from your existing code
from numpy.fft import fft2, ifft2, fftshift, ifftshift

def to_kspace(image: np.ndarray) -> np.ndarray:
    """Convert image to k-space using 2D FFT."""
    return fftshift(fft2(ifftshift(image)))

def from_kspace(kspace: np.ndarray) -> np.ndarray:
    """Reconstruct image from k-space using inverse 2D FFT."""
    return np.abs(fftshift(ifft2(ifftshift(kspace))))

def create_random_shepp_logan(size: int = 256, num_ellipses: int = None) -> np.ndarray:
    """Create a randomized Shepp-Logan phantom with variable ellipses.
    
    Args:
        size: Size of the phantom (size x size).
        num_ellipses: Number of ellipses to add (random if None).
        
    Returns:
        2D phantom image as numpy array.
    """
    if num_ellipses is None:
        num_ellipses = np.random.randint(3, 8)
    
    x = np.linspace(-1, 1, size)
    y = np.linspace(-1, 1, size)
    X, Y = np.meshgrid(x, y)
    
    phantom = np.zeros((size, size))
    
    # Background ellipse (always present)
    bg_width = np.random.uniform(0.6, 0.8)
    bg_height = np.random.uniform(0.8, 0.95)
    mask = ((X/bg_width)**2 + (Y/bg_height)**2) <= 1
    phantom[mask] = 1.0
    
    # Add random ellipses
    for i in range(num_ellipses - 1):
        # Random position
        cx = np.random.uniform(-0.5, 0.5)
        cy = np.random.uniform(-0.5, 0.5)
        
        # Random size
        width = np.random.uniform(0.1, 0.4)
        height = np.random.uniform(0.1, 0.4)
        
        # Random angle
        angle = np.random.uniform(0, np.pi)
        
        # Rotation matrix
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)
        X_rot = cos_a * (X - cx) + sin_a * (Y - cy)
        Y_rot = -sin_a * (X - cx) + cos_a * (Y - cy)
        
        # Create ellipse
        mask = ((X_rot/width)**2 + (Y_rot/height)**2) <= 1
        
        # Random intensity
        intensity = np.random.uniform(0.3, 0.9)
        phantom[mask] = intensity
    
    return phantom

def create_undersampling_mask_with_acs(
        shape: Tuple[int, int],
        acceleration: int = 4,
        acs_lines: int = 24
    ) -> np.ndarray:
    """Create undersampling mask with guaranteed ACS region for GRAPPA.
    
    Args:
        shape: Shape of the mask (height, width).
        acceleration: Acceleration factor.
        acs_lines: Number of fully sampled lines in center (ACS region).
        
    Returns:
        Binary mask where True indicates sampled locations.
    """
    mask = np.zeros(shape, dtype=bool)
    ny, nx = shape
    
    # Center ACS region
    center = ny // 2
    acs_start = center - acs_lines // 2
    acs_end = acs_start + acs_lines
    mask[acs_start:acs_end, :] = True
    
    # Random sampling outside ACS
    # Sample every R-th line on average, but randomly
    for i in range(ny):
        if i < acs_start or i >= acs_end:
            if np.random.rand() < 1.0 / acceleration:
                mask[i, :] = True
    
    return mask

def generate_dataset(
        num_train: int = 1000,
        num_test: int = 200,
        size: int = 256,
        acceleration: int = 4,
        save_path: str = './mri_dataset.h5'
    ):
    """Generate training and test datasets for MRI reconstruction.
    
    Args:
        num_train: Number of training samples.
        num_test: Number of test samples.
        size: Size of images.
        acceleration: Acceleration factor for undersampling.
        save_path: Path to save the dataset.
    """
    Path(save_path).parent.mkdir(exist_ok=True)
    
    with h5py.File(save_path, 'w') as f:
        # Create groups
        train_grp = f.create_group('train')
        test_grp = f.create_group('test')
        
        # Create datasets
        for grp, num_samples, grp_name in [(train_grp, num_train, 'train'), 
                                           (test_grp, num_test, 'test')]:
            # Preallocate arrays
            phantoms = grp.create_dataset('phantoms', (num_samples, size, size), dtype='float32')
            kspace_full = grp.create_dataset('kspace_full', (num_samples, size, size), dtype='complex64')
            kspace_undersampled = grp.create_dataset('kspace_undersampled', (num_samples, size, size), dtype='complex64')
            masks = grp.create_dataset('masks', (num_samples, size, size), dtype='bool')
            
            print(f"Generating {grp_name} dataset...")
            for i in tqdm(range(num_samples)):
                # Create phantom
                phantom = create_random_shepp_logan(size)
                phantoms[i] = phantom
                
                # Convert to k-space
                kspace = to_kspace(phantom)
                kspace_full[i] = kspace
                
                # Create undersampling mask
                mask = create_undersampling_mask_with_acs(
                    kspace.shape, 
                    acceleration=acceleration,
                    acs_lines=24
                )
                masks[i] = mask
                
                # Apply undersampling
                kspace_undersampled[i] = kspace * mask
        
        # Save metadata
        f.attrs['size'] = size
        f.attrs['acceleration'] = acceleration
        f.attrs['num_train'] = num_train
        f.attrs['num_test'] = num_test
    
    print(f"Dataset saved to {save_path}")

def visualize_samples(dataset_path: str, num_samples: int = 4):
    """Visualize some samples from the dataset."""
    with h5py.File(dataset_path, 'r') as f:
        train_data = f['train']
        
        fig, axes = plt.subplots(num_samples, 4, figsize=(12, 3*num_samples))
        
        for i in range(num_samples):
            # Phantom
            axes[i, 0].imshow(train_data['phantoms'][i], cmap='gray')
            axes[i, 0].set_title(f'Phantom {i}')
            axes[i, 0].axis('off')
            
            # Full k-space
            kspace = train_data['kspace_full'][i]
            axes[i, 1].imshow(np.log(np.abs(kspace) + 1e-8), cmap='gray')
            axes[i, 1].set_title('Full k-space (log)')
            axes[i, 1].axis('off')
            
            # Mask
            axes[i, 2].imshow(train_data['masks'][i], cmap='gray')
            axes[i, 2].set_title('Sampling mask')
            axes[i, 2].axis('off')
            
            # Undersampled reconstruction
            kspace_us = train_data['kspace_undersampled'][i]
            recon = from_kspace(kspace_us)
            axes[i, 3].imshow(recon, cmap='gray')
            axes[i, 3].set_title('Zero-filled recon')
            axes[i, 3].axis('off')
        
        plt.tight_layout()
        plt.show()

# %%
if __name__ == "__main__":
    # Generate dataset
    generate_dataset(
        num_train=2000,
        num_test=200,
        size=256,
        acceleration=4,
        save_path='./mri_dataset.h5'
    )
    
    # Visualize some samples
    visualize_samples('./mri_dataset.h5', num_samples=4)