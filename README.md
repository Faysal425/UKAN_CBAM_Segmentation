# UKAN-CBAM: U-shaped Kolmogorov-Arnold Network with CBAM for Binary Segmentation

This repository contains the implementation of UKAN-CBAM, a U-shaped Kolmogorov-Arnold Network with Convolutional Block Attention Module (CBAM) for binary image segmentation tasks.

## Architecture Overview

UKAN-CBAM combines:
- **Kolmogorov-Arnold Networks (KAN)**: A neural network architecture based on the Kolmogorov-Arnold representation theorem
- **CBAM (Convolutional Block Attention Module)**: Channel and spatial attention mechanisms
- **U-Net architecture**: Encoder-decoder structure with skip connections

## Installation

### Using pip
```bash
pip install -r requirements.txt
```

### Using conda
```bash
conda env create -f environment.yml
conda activate ukan-cbam
```

## Dataset Structure

Organize your dataset as follows:
```
data/
├── train/
│   ├── images/
│   └── masks/
├── val/
│   ├── images/
│   └── masks/
└── test/
    ├── images/
    └── masks/
```

## Training

### Basic training
```bash
python src/ukan_cbam/train.py --config configs/default.yaml
```

## Configuration

The model can be configured using YAML files in the `configs/` directory. Key parameters include:

- **Model architecture**: embed_dims, depths, add_cbam, no_kan
- **Training**: epochs, learning rate, batch size
- **Augmentation**: various image augmentation techniques
- **Loss functions**: Dice, BCE, IoU, Focal, etc.

## Model Components

### CBAM (Convolutional Block Attention Module)
- **Channel Attention**: Focuses on 'what' is meaningful
- **Spatial Attention**: Focuses on 'where' is meaningful

### KAN Layers
- **KANLinear**: Kolmogorov-Arnold Network linear layers
- **B-spline basis functions**: For function approximation
- **Adaptive grid update**: Dynamic grid adjustment during training

## Citation

```bibtex
@article{AHAMED2026103352,
    title = {Rethinking U-Net architecture in medical imaging: Advancing the efficient and interpretable UKAN-CBAM framework for colorectal polyp segmentation},
    journal = {Artificial Intelligence in Medicine},
    pages = {103352},
    year = {2026},
    issn = {0933-3657},
    doi = {https://doi.org/10.1016/j.artmed.2026.103352},
    url = {https://www.sciencedirect.com/science/article/pii/S0933365726000047},
    author = {Md. Faysal Ahamed and Fariya Bintay Shafi and Md. Rabiul Islam and Md. Fahmidun Nabi and Julfikar Haider},
    keywords = {Colorectal cancer, Colorectal polyps, KANs (Kolmogorov-Arnold networks), UKAN (U-Net with KAN), CBAM (Convolutional block attention module), UKAN-CBAM, Kvasir-SEG}
}
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
