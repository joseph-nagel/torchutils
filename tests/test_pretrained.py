'''Tests for the pretrained module.'''

import pytest
import torch
from torch.utils.data import TensorDataset, DataLoader
from torchvision import models

from torchutils.pretrained import create_feature_extractor, extract_features


@pytest.fixture
def data_image_loader():
    '''Create a data loader with random images.'''
    torch.manual_seed(0)

    no_samples = 10
    image_shape = (224, 224)

    X = torch.randn(no_samples, 3, *image_shape)
    y = torch.randint(2, size=(no_samples,))

    data_set = TensorDataset(X, y)
    data_loader = DataLoader(data_set, batch_size=32, shuffle=True)

    return data_loader


@pytest.mark.parametrize('model_constructor, expected_feature_shape', [
    (models.alexnet, (256, 6, 6)),
    (models.vgg16, (512, 7, 7)),
    (models.resnet18, (512, 7, 7)),
    (models.densenet121, (1024, 7, 7))
])
def test_feature_extraction(data_image_loader,
                            model_constructor,
                            expected_feature_shape):
    '''Test pretrained feature extraction.'''
    data_loader = data_image_loader
    pretrained_model = model_constructor(pretrained=False)

    feature_extractor, feature_shape = create_feature_extractor(pretrained_model)
    features, labels = extract_features(feature_extractor, data_loader)

    assert features.shape[1:] == expected_feature_shape

