# MRI Reconstruction Project

This repository implements and compares MRI reconstruction techniques from classical interpolation methods to modern deep learning approaches for accelerated MRI from undersampled k-space data.

## Repository Structure

```
├── lightning_logs/
├── src/
│   ├── ImageReconstruction/
│   │   ├── __pycache__/
│   │   ├── different_interpolation_methods/
│   │   │   ├── Figures/
│   │   │   ├── different_interpolation_methods.py
│   │   │   └── README.md
│   │   └── first_reconstruction/
│   │       ├── first_reconstruction.png
│   │       ├── first_reconstruction.py
│   │       └── README.md
│   ├── tv_vs_unet/
│   │   ├── Plots/
│   │   ├── compare_methods.py
│   │   ├── compare_cascased_with_tv.py
│   │   ├── data_generation.py
│   │   ├── train_unet.py
│   │   ├── train_cascaded_refinement.py
│   │   └── README.md
│   └── __init__.py
```

## Project Components

### 🚀 **Cascaded Deep Learning MRI Reconstruction** (`tv_vs_unet/`)

Advanced two-stage deep learning approach that combines k-space and image-space reconstruction for superior MRI recovery from undersampled data.

**Key Innovation:**
- **Cascaded Architecture**: K-space U-Net followed by image-space refinement CNN
- **Surprising Discovery**: TV denoising can compete with U-Net on SSIM metrics due to smoothness bias
- **Data-Consistent K-space Stage**: Adaptive U-Net with built-in measured value replacement
- **Residual Image Refinement**: CNN that removes artifacts while preserving structural details
- **Advanced Training Framework**: PyTorch Lightning with mixed precision, experiment tracking via Weights & Biases

**Results Highlights:**
- Cascaded model achieves highest PSNR/SSIM and lowest RMSE across all methods
- Superior and more stable performance compared to individual approaches
- Effective combination of frequency domain data consistency with spatial domain priors

**📖 [Detailed Documentation](src/tv_vs_unet/README.md)**

### 📁 **Classical Reconstruction Methods** (`ImageReconstruction/`)

Comprehensive exploration of traditional MRI reconstruction techniques and fundamental concepts.

#### **📁 Fundamentals** (`first_reconstruction/`)
- **Basic Concepts**: Zero-filled reconstruction and k-space fundamentals
- **Foundation Learning**: Introduction to MRI reconstruction principles
- **📖 [Getting Started Guide](src/ImageReconstruction/first_reconstruction/README.md)**

#### **📁 Advanced Classical Methods** (`different_interpolation_methods/`)
- **Comprehensive Comparison**: Linear/spline k-space interpolation, radial interpolation, and low-pass filtering
- **Multiple Phantom Types**: Various undersampling patterns and scenarios
- **Quantitative Analysis**: Performance metrics and frequency domain analysis
- **📖 [Classical Methods Analysis](src/ImageReconstruction/different_interpolation_methods/README.md)**

## Technical Stack
- **Deep Learning**: PyTorch, PyTorch Lightning
- **Scientific Computing**: NumPy, SciPy, scikit-image
- **Data & Logging**: HDF5, Weights & Biases
- **Visualization**: Matplotlib, PIL

## Getting Started

1. **🏁 Start Here**: Begin with [first_reconstruction](src/ImageReconstruction/first_reconstruction/) for fundamental MRI reconstruction concepts
2. **📚 Classical Foundations**: Explore [different_interpolation_methods](src/ImageReconstruction/different_interpolation_methods/) for comprehensive classical approaches
3. **🚀 Modern Techniques**: Dive into [tv_vs_unet](src/tv_vs_unet/) for state-of-the-art cascaded deep learning reconstruction

Each component contains detailed documentation with implementation specifics, usage instructions, and comprehensive result interpretations.

## Future Research Directions

### 🤖 **K-space Agent**
Develop an agent that can adaptively determine optimal k-space sampling patterns and reconstruction strategies for different imaging scenarios. This agent would learn to balance acquisition speed with reconstruction quality based on the specific imaging context.