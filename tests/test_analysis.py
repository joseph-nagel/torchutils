'''Tests for the analysis module.'''

import pytest
import numpy as np
import torch
from sklearn.metrics import confusion_matrix as confusion_matrix

from torchutils.analysis import conf_mat as conf_mat


@pytest.fixture
def data_y_true_y_top1():
    '''Sample ground truth labels and top-class predictions.'''
    torch.manual_seed(0)

    no_classes = 10
    no_samples = 1000

    y_true = torch.randint(no_classes, size=(no_samples,))
    y_top1 = torch.randint(no_classes, size=(no_samples,))

    return y_true, y_top1


def test_conf_mat(data_y_true_y_top1):
    '''Test correctness against a reference implementation.'''
    y_true, y_top1 = data_y_true_y_top1

    confmat = conf_mat(y_true, y_top1).numpy()
    reference = confusion_matrix(y_true.numpy(), y_top1.numpy())

    assert np.allclose(confmat, reference)

