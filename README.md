# MRI Reconstruction Project

This repository implements and compares MRI reconstruction techniques from classical interpolation methods to modern deep learning approaches for accelerated MRI from undersampled k-space data.

## Repository Structure

### 📁 `src/grappa_vs_unet/`
**Deep Learning MRI Reconstruction**

- **`data_generation.py`**: Generates synthetic Shepp-Logan phantoms with realistic undersampling patterns
- **`train_unet.py`**: Adaptive U-Net with acceleration-aware architecture and built-in data consistency
- **[📖 README.md](src/grappa_vs_unet/README.md)**: Detailed implementation details and results

**Key Components:**
- Synthetic dataset generation with configurable acceleration factors
- U-Net architecture with adaptive kernel sizes based on acceleration
- k-space domain training with data consistency enforcement
- PyTorch Lightning training framework with experiment tracking

### 📁 `src/ImageReconstruction/`
**Classical Reconstruction Methods**

#### `first_reconstruction/`
- **`first_reconstruction.py`**: Basic zero-filled reconstruction and k-space fundamentals
- **[📖 README.md](src/ImageReconstruction/first_reconstruction/README.md)**: Introduction to MRI reconstruction concepts

#### `different_interpolation_methods/`
- **`different_interpolation_methods.py`**: Comprehensive comparison of classical techniques including linear/spline k-space interpolation, radial interpolation, and low-pass filtering
- **[📖 README.md](src/ImageReconstruction/different_interpolation_methods/README.md)**: Detailed analysis of classical methods

**Key Components:**
- Multiple interpolation strategies in k-space domain
- Various phantom types and undersampling patterns
- Quantitative metrics and frequency domain analysis
- Performance comparison across different scenarios

## Technical Stack
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Scientific Computing**: NumPy, SciPy, scikit-image
- **Data & Logging**: HDF5, Weights & Biases

## Getting Started
1. **Basics**: Start with [first_reconstruction](src/ImageReconstruction/first_reconstruction/) for fundamental concepts
2. **Classical Methods**: Explore [different_interpolation_methods](src/ImageReconstruction/different_interpolation_methods/) for comprehensive classical approaches
3. **Deep Learning**: Dive into [grappa_vs_unet](src/grappa_vs_unet/) for modern reconstruction techniques

Each folder contains detailed README files with specific implementation details, usage instructions, and result interpretations.

## Future Ideas
- Add another U-Net that learns to reconstruct the image in image space with the real image as target. I feel like there are constraints in both spaces (Image, and k-space) that complement each other
- Make a reinforcement learning project, where the actor learns to choose which k-space lines to sample next to maximise the psnr.