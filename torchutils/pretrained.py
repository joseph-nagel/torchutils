'''
Pretrained models.

Summary
-------
A module to create and use feature extractors based on pretrained models.
The interface is unified over multiple model architectures shipped by torchvision.

Notes
-----
Note that ImageNet-pretrained models usually expect images of shape (3, 224, 224).
A noteworthy exception is Inception3 with (3, 299, 299) that also features auxiliary outputs.
The torchvision implementations also accept images with higher spatial sizes.
Here, the dimensionality gets standardized directly before the fully connected layers.
For all pretrained models, inputs should be normalized to zero mean and unit variance.

'''

import numpy as np
import torch
import torch.nn as nn
from torchvision.models import AlexNet, VGG, ResNet, Inception3

# TODO: check correctness for ResNet, add Inception3
def create_feature_extractor(model_architecture,
                             no_layers=None,
                             input_shape=None,
                             is_pretrained=True,
                             is_frozen=True):
    '''
    Create a pretrained feature extractor.

    Summary
    -------
    A feature extractor is created from a predefined model architecture.
    At the moment, AlexNet, VGG, and ResNet models are supported.
    The shape of the output features is empirically determined.

    Parameters
    ----------
    model_architecture : PyTorch module or model constructor
        Predefined model architecture.
    no_layers : int or None
        Last layer used for the features.
    input_shape : tuple or None
        Model input shape. If not given, ImageNet defaults are used.
    is_pretrained : bool
        Determines whether pretrained weights are loaded.
    is_frozen : bool
        Determines whether weights are frozen.

    '''

    # initialize model
    if isinstance(model_architecture, (AlexNet, VGG, ResNet)):
        pretrained_model = model_architecture # is already a model instance
    else:
        pretrained_model = model_architecture(pretrained=is_pretrained) # is only a model constructor

    # freeze parameters
    if is_frozen:
        freeze_parameters(pretrained_model)

    # input shape
    if input_shape is None:
        if not isinstance(pretrained_model, Inception3):
            input_shape = (3, 224, 224)
        else:
            input_shape = (3, 299, 299)
    if len(input_shape) == 2:
        input_shape = (3,) + input_shape

    # last layer index
    last_layer_idx = None if no_layers is None else no_layers + 1

    # create features
    if isinstance(pretrained_model, (AlexNet, VGG)):
        # feature_list = list(pretrained_model.children())[:-2][0]
        feature_list = list(pretrained_model.features.children())
    elif isinstance(pretrained_model, ResNet):
        feature_list = list(pretrained_model.children())[:-2]
        feature_list.append(nn.AvgPool2d((7,7)))
    feature_extractor = nn.Sequential(*feature_list[0:last_layer_idx])

    # feature shape
    feature_shape = get_output_shape(feature_extractor, input_shape)
    # no_features = np.prod(feature_shape)

    return feature_extractor, feature_shape

def freeze_parameters(model):
    '''Freeze the weights of a model.'''
    for param in model.parameters():
        param.requires_grad = False

def get_output_shape(model, input_shape):
    '''
    Return the ouput shape of a model.

    Summary
    -------
    The model ouput shape is empirically determined for input tensors of a certain shape.
    Its values are randomly sampled from a standard normal distribution.
    Very generally, for a given input tensor with shape (no_samples, *input_shape),
    the model predicts an (no_samples, *output_shape)-shaped output.
    The shape of this output, without the sample size, is returned.

    '''

    model.eval()
    with torch.no_grad():
        predictions = model(torch.randn(1, *input_shape))
    output_shape = tuple(predictions.shape[1:])
    return output_shape

def extract_features(feature_extractor, data_loader, expand=None, as_array=False):
    '''
    Extract features given a model and a data loader.

    Summary
    -------
    Extracts features from all images generated by the data loader.
    After they are computed in batches, they are eventually concatenated
    and returned as either PyTorch tensors or Numpy arrays.

    Parameters
    ----------
    feature_extractor : PyTorch module.
        Feature extraction model.
    data_loader : PyTorch DataLoader.
        Data loader instance.
    expand : tuple or None
        If given, input tensors are expanded accordingly.
    as_array : bool
        Determines whether outputs are returned as Numpy arrays.

    '''

    features_list = []
    labels_list = []
    feature_extractor.eval()
    with torch.no_grad():
        for images, labels in data_loader:
            if expand is not None:
                images = images.expand(*expand)
            features = feature_extractor(images)
            features_list.append(features)
            labels_list.append(labels)
    if as_array: # Numpy arrays
        features = np.concatenate([tensor.numpy() for tensor in features_list], axis=0)
        labels = np.concatenate([tensor.numpy() for tensor in labels_list], axis=0)
    else: # PyTorch tensors
        features = torch.cat([tensor for tensor in features_list], dim=0)
        labels = torch.cat([tensor for tensor in labels_list], dim=0)
    return features, labels

