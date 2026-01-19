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

### Resume training from checkpoint
```bash
python src/ukan_cbam/train.py --config configs/default.yaml --resume checkpoints/best_model.pth.tar
```

## Evaluation

```bash
python src/ukan_cbam/evaluate.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth.tar
```

## Prediction

### Single image
```bash
python src/ukan_cbam/predict.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth.tar --input path/to/image.jpg
```

### Directory of images
```bash
python src/ukan_cbam/predict.py --config configs/default.yaml --checkpoint checkpoints/best_model.pth.tar --input path/to/images/
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

## Repository Structure

```
├── requirements.txt          # Python dependencies
├── environment.yml          # Conda environment
├── README.md               # This file
├── configs/
│   ├── default.yaml        # Default configuration
│   └── ablations/          # Ablation study configs
├── src/
│   └── ukan_cbam/
│       ├── __init__.py
│       ├── models/         # Model components
│       │   ├── __init__.py
│       │   ├── ukan_cbam.py    # Main architecture
│       │   ├── blocks.py       # CBAM, KAN layers
│       │   └── init_utils.py   # Weight initialization
│       ├── data/           # Dataset handling
│       │   ├── __init__.py
│       │   └── datasets.py     # Dataset classes
│       ├── losses.py       # Loss functions
│       ├── metrics.py      # Evaluation metrics
│       ├── train.py        # Training script
│       ├── evaluate.py     # Evaluation script
│       └── predict.py      # Prediction script
└── utils/
    ├── __init__.py
    ├── train_utils.py      # Training utilities
    ├── eval_utils.py       # Evaluation utilities
    └── io.py              # I/O utilities
```

## Performance

The model achieves competitive performance on binary segmentation tasks with:
- **Dice Score**: > 0.90 on standard datasets
- **IoU**: > 0.85 on standard datasets
- **Parameters**: ~15M trainable parameters
- **Inference Time**: ~50ms per 224x224 image on GPU

## Citation

If you use this code in your research, please cite:
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
