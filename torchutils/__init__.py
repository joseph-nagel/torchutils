'''
PyTorch utilities for Keras-like convenience.

Summary
-------
This package provides high-level utilities for PyTorch programming.
In the first place, model training and testing is facilitated.
Moreover, the unified import and use of pretrained models is supported.
Some auxiliary functions related to data handling are implemented.

Modules
-------
analysis : Analysis tools.
classification : Classification problems.
data : Data processing.
loss : Loss functions.
pretrained : Pretrained models.
tools : General tools.

'''

from . import (
    analysis,
    classification,
    data,
    loss,
    pretrained,
    tools
)

from .analysis import confusion_matrix

from .classification import Classification

from .data import (
    mean_std_over_dataset,
    SingleTensorDataset,
    GaussianNoise,
    BalancedSampler,
    image2tensor,
    tensor2image
)

from .loss import HingeLoss, FocalLoss

from .pretrained import create_feature_extractor, extract_features

from .tools import conv_out_shape

