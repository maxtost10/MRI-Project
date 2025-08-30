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
│   │   ├── data_generation.py
│   │   ├── train_unet.py
│   │   └── README.md
│   └── __init__.py
```

## Project Components

### 📁 **Deep Learning MRI Reconstruction** (`tv_vs_unet/`)

Advanced deep learning approach implementing an adaptive U-Net architecture specifically designed for MRI reconstruction with comprehensive method comparison.

**Key Features:**
- **Synthetic Data Generation**: Creates realistic MRI phantom data with controlled undersampling patterns
- **Adaptive U-Net Architecture**: Acceleration-aware design with larger kernels (7×7) for wider spatial relationship capture
- **Built-in Data Consistency**: Enforces consistency between predicted and measured k-space data
- **Comprehensive Comparison**: Evaluates U-Net against Total Variance Minimization and zero-filled reconstruction methods
- **Advanced Training Framework**: PyTorch Lightning with mixed precision, experiment tracking via Weights & Biases

**Results Highlights:**
- Achieves 1+ dB PSNR improvement over classical methods
- SSIM improvements from 0.635 to 0.733 indicating better structural preservation
- Superior artifact removal and noise reduction across diverse phantom geometries

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
3. **🚀 Modern Techniques**: Dive into [tv_vs_unet](src/tv_vs_unet/) for state-of-the-art deep learning reconstruction

Each component contains detailed documentation with implementation specifics, usage instructions, and comprehensive result interpretations.

## Future Research Directions

### 🔬 **Dual-Domain Learning**
Implement a U-Net that learns to reconstruct additionally in image space with real images as targets. This approach would leverage complementary constraints in both k-space and image domains for potentially superior reconstruction quality.

### 🎯 **Reinforcement Learning for Adaptive Sampling**
Develop an RL framework where an agent learns optimal k-space sampling strategies to maximize reconstruction quality (PSNR). This could lead to intelligent, content-aware undersampling patterns that adapt to specific anatomy or pathology.