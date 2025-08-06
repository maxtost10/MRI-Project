# First reconstruction
A small script that creates a shepp logan phantom, transforms it to k-space, removes random higher $k_y$-lines and transforms it back to the image space.

# Functions

## create_shepp_logan_phantom(size: int=256):
Creates a square of black background with two concentric circles. The inner one is less bright than the outer one.

## to_kspace(image: np.array)
Applies the 2d fast fourier transform and shifts the frequencies such that frequency=0 is at the center of the transform.

## from_kspace(kspace: np.array)
Applies the 2d inverse fast fourier transform, assuming that frequency=0 is at the center of the k-space data.

## create_random_mask(shape: array (2d), acceleration: int=4)
Creates a mask that is supposed to mimic undersampling. The mask is always the same for same y-values when an x is chosen. For x-values, 8% in the center are set to 1 (which means that this region would be scanned) and from the outside region, 1/acceleration (=25% for acceleration=4) x values will be set to 1 randomly.

# Example Use

## Create Synthetic Data
```python
phantom = create_shepp_logan_phantom()

plt.figure(figsize=(10, 5))
plt.subplot(121)
plt.imshow(phantom, cmap='gray')
plt.title('Original Phantom')
plt.colorbar()
```

## Transform to k-space
```python
kspace = to_kspace(phantom)

plt.figure(figsize=(10, 5))
plt.subplot(122)
plt.imshow(np.log(np.abs(kspace) + 1e-9), cmap='gray')
plt.title('K-Space (log magnitude)')
plt.colorbar()
plt.show()
```

## Apply random mask to higher frequencies
```python
mask = create_random_mask(kspace.shape, acceleration=3)
undersampled_kspace = kspace * mask
zero_filled_recon = from_kspace(undersampled_kspace)
```

## Analyse reconstruction
```python
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
```