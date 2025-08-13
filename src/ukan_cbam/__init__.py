"""UKAN-CBAM: U-shaped Kolmogorov-Arnold Network with CBAM for binary segmentation."""

__version__ = "1.0.0"
__author__ = "Your Name"
__email__ = "your.email@example.com"

from .models.ukan_cbam import UKAN_CBAM
from .models.blocks import CBAM, KANLinear, KANLayer

__all__ = [
    "UKAN_CBAM",
    "CBAM",
    "KANLinear", 
    "KANLayer"
]
