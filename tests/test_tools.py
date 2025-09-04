'''Tests for the tools module.'''

import itertools

import pytest
import torch
import torch.nn as nn

from torchutils.tools import conv_out_shape


input_shapes = [(10, 10), (100, 100)]
kernel_sizes = [(3, 3), (5, 5)]
strides = [1]
paddings = [0, 1, 2]
dilations = [1]

cartesian_product = [elem for elem in itertools.product(
    input_shapes,
    kernel_sizes,
    strides,
    paddings,
    dilations
)]


@pytest.fixture(params=cartesian_product)
def data_conv_model_and_input(request):
    '''Create convolutional layer and input tensor.'''

    torch.manual_seed(0)

    input_shape, kernel_size, stride, padding, dilation = request.param

    model = nn.Conv2d(
        in_channels=1,
        out_channels=1,
        kernel_size=kernel_size,
        stride=stride,
        padding=padding,
        dilation=dilation
    )

    X = torch.randn(1, model.in_channels, *input_shape)

    return model, X


def test_conv_out_shape(data_conv_model_and_input):
    '''Test the predicted output shape after the convolution.'''

    model, X = data_conv_model_and_input

    y = model(X)

    actual_out_shape = y.shape[2:]

    predicted_out_shape = conv_out_shape(
        input_shape=X.shape[2:],
        kernel_size=model.kernel_size,
        stride=model.stride,
        padding=model.padding,
        dilation=model.dilation
    )

    assert predicted_out_shape == actual_out_shape
