"""
Amyloid plaque morphology classifier.

A two-part pipeline for classifying amyloid plaque morphology in fluorescence
microscopy:

  1. A ResNet50 CNN filters segmented objects into plaque vs not-plaque (cnn.py,
     needs PyTorch).
  2. Engineered morphology features (features.py) feed a Linear Discriminant
     Analysis (classifier.py) that assigns Diffuse / DenseCore / Compact and
     places each plaque on a 2D morphology continuum.

Whole-image handling (segmentation, tile extraction, overlays) is in
segmentation.py and pipeline.py. synthetic.py generates stand-in data so
everything runs without any real microscopy.
"""

from .features import compute_features, compute_features_batch, FEATURE_NAMES, N_FEATURES
from .classifier import (train_lda, save_bundle, load_bundle, classify_features,
                        PLAQUE_CLASSES)

__all__ = [
    'compute_features', 'compute_features_batch', 'FEATURE_NAMES', 'N_FEATURES',
    'train_lda', 'save_bundle', 'load_bundle', 'classify_features', 'PLAQUE_CLASSES',
]
