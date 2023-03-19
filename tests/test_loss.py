'''Tests for the loss module.'''

import pytest
import numpy as np
import torch
from sklearn.metrics import log_loss

from torchutils.loss import HingeLoss, FocalLoss


@pytest.fixture(params=[1, 10, 100, 1000])
def data_y_true_y_pred(request):
    '''Sample ground truth labels and predicted logits.'''
    torch.manual_seed(0)

    no_samples = request.param
    shape = (no_samples,)

    y_true = torch.randint(2, size=shape)
    y_pred = torch.randn(*shape)

    return y_true, y_pred


class TestHingeLoss:
    '''Tests of the hinge loss.'''

    def test_hinge_loss_correctness(self):
        '''Test the correctness at selected points.'''
        hinge_loss = HingeLoss()

        y_true = torch.tensor(1)
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(2.), y_true=y_true), torch.tensor(0.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(1.), y_true=y_true), torch.tensor(0.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(-0.), y_true=y_true), torch.tensor(1.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(-1.), y_true=y_true), torch.tensor(2.))

        y_true = torch.tensor(-1)
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(-2.), y_true=y_true), torch.tensor(0.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(-1.), y_true=y_true), torch.tensor(0.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(0.), y_true=y_true), torch.tensor(1.))
        assert torch.isclose(hinge_loss(y_pred=torch.tensor(1.), y_true=y_true), torch.tensor(2.))

    def test_hinge_loss_reduction(self, data_y_true_y_pred):
        '''Test the reduction over multiple datapoints.'''
        y_true, y_pred = data_y_true_y_pred
        y_true[y_true == 0] = -1

        hinge_loss_mean = HingeLoss(reduction='mean')
        hinge_loss_sum = HingeLoss(reduction='sum')

        loss = hinge_loss_sum(y_pred, y_true)
        reference = y_true.numel() * hinge_loss_mean(y_pred, y_true)

        assert torch.isclose(loss, reference)

    def test_hinge_loss_squared(self, data_y_true_y_pred):
        '''Test the squared hinge loss.'''
        y_true, y_pred = data_y_true_y_pred
        y_true[y_true == 0] = -1

        y_true = y_true.view(-1)
        y_pred = y_pred.view(-1)

        hinge_loss_nonsquared = HingeLoss(squared=False)
        hinge_loss_squared = HingeLoss(squared=True)

        losses = torch.tensor([hinge_loss_squared(y_pred[idx], y_true[idx]) for idx in range(len(y_true))])
        references = torch.tensor([hinge_loss_nonsquared(y_pred[idx], y_true[idx])**2 for idx in range(len(y_true))])

        assert torch.allclose(losses, references)


class TestFocalLoss:
    '''Tests of the focal loss.'''

    def test_focal_loss_correctness(self, data_y_true_y_pred):
        '''Test the correctness against a binary cross entropy.'''
        y_true, y_pred = data_y_true_y_pred

        focal_loss = FocalLoss()

        loss = focal_loss(y_pred, y_true).numpy()
        reference = log_loss(y_true.numpy().ravel(), torch.sigmoid(y_pred).numpy().ravel(), labels=[0,1])

        assert np.isclose(loss, reference)

    def test_focal_loss_reduction(self, data_y_true_y_pred):
        '''Test the reduction over multiple datapoints.'''
        y_true, y_pred = data_y_true_y_pred

        focal_loss_mean = FocalLoss(reduction='mean')
        focal_loss_sum = FocalLoss(reduction='sum')

        loss = focal_loss_sum(y_pred, y_true)
        reference = y_true.numel() * focal_loss_mean(y_pred, y_true)

        assert torch.isclose(loss, reference)

    def test_focal_loss_posweight(self, data_y_true_y_pred):
        '''Test varying the class weights.'''
        y_true, y_pred = data_y_true_y_pred

        y_ones = torch.ones_like(y_true)
        y_zeros = torch.zeros_like(y_true)

        focal_loss_noweight = FocalLoss()
        focal_loss_weighted = FocalLoss(pos_weight=2.)

        loss_ones = focal_loss_weighted(y_pred, y_ones)
        reference_ones = focal_loss_weighted.pos_weight * focal_loss_noweight(y_pred, y_ones)

        loss_zeros = focal_loss_weighted(y_pred, y_zeros)
        reference_zeros = focal_loss_noweight(y_pred, y_zeros)

        assert torch.isclose(loss_ones, reference_ones)
        assert torch.isclose(loss_zeros, reference_zeros)

    def test_focal_loss_focalgamma(self, data_y_true_y_pred):
        '''Test the plausibility for differing focal parameters.'''
        y_true, y_pred = data_y_true_y_pred

        focal_loss_gamma0 = FocalLoss(focal_gamma=0.)
        focal_loss_gamma1 = FocalLoss(focal_gamma=1.)

        loss_gamma0 = focal_loss_gamma0(y_pred, y_true)
        loss_gamma1 = focal_loss_gamma1(y_pred, y_true)

        assert loss_gamma1 < loss_gamma0

