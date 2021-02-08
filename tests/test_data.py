'''Tests for the data module.'''

import pytest
import numpy as np
import torch
from torch.utils.data import TensorDataset
from torchutils.data import mean_std_over_dataset, image2tensor, tensor2image

@pytest.mark.parametrize('no_samples', [100, 1000])
@pytest.mark.parametrize('feature_shape', [(), (1,), (10,), (10,10)])
def test_mean_std_over_dataset(no_samples, feature_shape):
    '''Test correctness of evaluating the mean and standard deviation.'''
    torch.manual_seed(0)
    X = torch.randn(no_samples, *feature_shape)
    y = torch.randint(2, size=(no_samples,))
    data_set = TensorDataset(X, y)
    mean, std = mean_std_over_dataset(data_set)
    ref_mean = X.numpy().mean()
    ref_std = X.numpy().std()
    assert np.isclose(mean, ref_mean, rtol=1e-02, atol=1e-03)
    assert np.isclose(std, ref_std, rtol=1e-02, atol=1e-03)

@pytest.mark.parametrize('shape', [(10,10), (10,10,3), (1,10,10,3)])
def test_image2tensor2image(shape):
    '''Test the transformation and back-transformation of an image.'''
    np.random.seed(0)
    image = np.random.randn(*shape)
    tensor = image2tensor(image)
    new_image = tensor2image(tensor)
    assert np.allclose(image.squeeze(), new_image.squeeze())

@pytest.mark.parametrize('shape', [(10,10), (3,10,10), (1,3,10,10)])
def test_tensor2image2tensor(shape):
    '''Test the transformation and back-transformation of a tensor.'''
    torch.manual_seed(0)
    tensor = torch.randn(*shape)
    image = tensor2image(tensor)
    new_tensor = image2tensor(image)
    assert np.allclose(tensor.squeeze(), new_tensor.squeeze())

