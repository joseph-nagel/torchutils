'''
General tools.

Summary
-------
This module serves as a collection of tools that have not been
deemed specific or relevant enough to be located elsewhere.
Nevertheless, they might be useful at times.

'''

import numpy as np

def moving_average(x, window=3, mode='full'):
    '''
    Calculate the moving average over an array.

    Summary
    -------
    This function computes the running mean of an array.
    Padding is performed for the 'left' side, not for the 'right'.

    Parameters
    ----------
    x : array
        Input array.
    window : int
        Window size.
    mode : {'full', 'last'}
        Determines whether the full rolling mean history
        or only its last element is returned.

    Returns
    -------
    running_mean : float
        Rolling mean.

    '''

    x = np.array(x)
    if mode == 'full':
        x_padded = np.pad(x, (window-1, 0), mode='constant', constant_values=x[0])
        running_mean = np.convolve(x_padded, np.ones((window,))/window, mode='valid')
    elif mode == 'last':
        if x.size >= window:
            running_mean = np.convolve(x[-window:], np.ones((window,))/window, mode='valid')[0]
        else:
            x_padded = np.pad(x, (window-x.size, 0), mode='constant', constant_values=x[0])
            running_mean = np.convolve(x_padded, np.ones((window,))/window, mode='valid')[0]
    return running_mean

def conv_out_shape(input_shape,
                   kernel_size,
                   stride=1,
                   padding=0,
                   dilation=1,
                   mode='floor'):
    '''
    Calculate the output shape of a convolutional layer.

    Summary
    -------
    This function returns the output tensor shape of a convolutional layer.
    One needs to pass the input shape and all relevant layer properties as arguments.
    The parameter convention of PyTorch's convolutional layer modules is adopted herein,
    e.g. see the documentation of the 'torch.nn.Conv2d' class.

    Parameters
    ----------
    input_shape : int or array-like
        Shape of the layer input tensor.
    kernel_size : int or array-like
        Size of the convolutional kernels.
    stride : int or array-like
        Stride parameter.
    padding : int or array-like
        Padding parameter.
    dilation : int or array-like
        Dilation parameter.
    mode : {'floor', 'ceil'}
        Determines whether to floor or to ceil.

    Returns
    -------
    output_shape : int or tuple
        Shape of the layer output tensor.

    Notes
    -----
    The same function can be used to determine the output size of pooling layers.
    Though, some care regarding the ceil/floor mode has to be taken.
    PyTorch's default behavior is to floor the output size.

    '''

    input_shape = np.array(input_shape)
    no_dims = input_shape.size

    kernel_size = make_array(kernel_size, no_dims)
    stride = make_array(stride, no_dims)
    padding = make_array(padding, no_dims)
    dilation = make_array(dilation, no_dims)

    if mode == 'floor':
        output_shape = np.floor((input_shape + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1).astype('int')
    elif mode == 'ceil':
        output_shape = np.ceil((input_shape + 2*padding - dilation*(kernel_size-1) - 1) / stride + 1).astype('int')

    if no_dims == 1:
        output_shape = int(output_shape)
    if no_dims >= 2:
        output_shape = tuple([int(output_shape[i]) for i in range(no_dims)])

    return output_shape

def make_array(x, no_dims):
    '''Transform a scalar into an array with equal entries.'''
    return np.array(x) if np.size(x) == no_dims else np.array([x for i in range(no_dims)])

