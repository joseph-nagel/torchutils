'''
Loss functions.

Summary
-------
The module is supposed to contain some less common loss functions.
Currently, this only involes the hinge loss for binary classification.
Even though it is lesser known in the deep learning community,
the hinge loss, or its squared variant, has interesting properties.
When used in conjunction with an l2 parameter penalty,
which acts as maximum-margin regularization term,
it promotes SVM-like soft-margin classifiers.

Notes
-----
It is remarked that penalization/regularization for the model weights
are usually specified during the initialization of a PyTorch optimizer.
Since it refers to a term in the loss, and not a variant of the optimizer,
that might seem somewhat counterintuitive at first sight.

'''

import torch
import torch.nn as nn

class HingeLoss(nn.Module):
    '''
    Hinge loss function.

    Summary
    -------
    A hinge loss for binary classification is implemented as a PyTorch module.
    The function expects {-1,1}-encoded ground thruth class labels.
    This is the most natural representation in this context.
    If labels are specified otherwise, they have to be transformed beforehand,
    e.g. from the {0,1}-representation often met in binary classification.

    '''

    def __init__(self, squared=False, reduction='mean'):
        super().__init__()
        self.squared = squared
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum

    def forward(self, y_pred, y_true):
        if not self.squared: # hinge loss
            loss = self.reduce(torch.clamp(1 - y_true.squeeze() * y_pred.squeeze(), min=0))
        else: # squared hinge loss
            loss = self.reduce(torch.clamp(1 - y_true.squeeze() * y_pred.squeeze(), min=0)**2)
        return loss

class FocalLoss(nn.Module):
    '''
    Focal loss function.

    Summary
    -------
    A weighted binary cross entropy loss is implemented.
    It can be used for binary classification or semantic segmentation.
    The formulation is a variant of the focal loss function introduced in [1].
    It is a generalization of the binary cross entropy.
    Class imbalance is addressed by a factor that correspondingly weights the contributions.
    Similarly, a focusing parameter determines in as how much easily classified samples
    are down-weighted in order to better focus on difficult-to-classify ones.

    Parameters
    ----------
    pos_weight : float, optional (default=1.)
        Balancing weight of the positive class (1) relative to the negative class (0).
        The parameter takes on the absolute value of the passed input.
    focal_gamma : float, optional (default=0.)
        Modulating parameter that determines the degree to which easily classified
        samples are down-weighted and difficult ones are focussed on.
        The parameter takes on the absolute value of the passed input.
    reduction : {'mean', 'sum'}, optional (default='mean')
        Determines the reduction mode, i.e. how the individual loss terms are summarized.

    Notes
    -----
    It is noted that the binary cross entropy expects {0,1}-encoded labels.
    Real-valued targets in [0,1] would lead to a questionable interpretation.
    A related discussion in the context of VAEs can be found in [2].

    References
    ----------
    [1] Lin et al, "Focal Loss for Dense Object Detection", IEEE Trans. Pattern Anal. Mach. Intell. 42(2), 2020.
        https://arxiv.org/abs/1708.02002
    [2] Loaiza-Ganem and Cunningham, "The continuous Bernoulli: fixing a pervasive error in variational autoencoders", NeurIPS 2019.
        https://papers.nips.cc/paper/9484-the-continuous-bernoulli-fixing-a-pervasive-error-in-variational-autoencoders

    '''

    def __init__(self, pos_weight=1., focal_gamma=0., reduction='mean'):
        super().__init__()
        self.pos_weight = abs(pos_weight)
        self.focal_gamma = abs(focal_gamma)
        if reduction == 'mean':
            self.reduce = torch.mean
        elif reduction == 'sum':
            self.reduce = torch.sum

    def forward(self, y_pred, y_true):
        '''
        Compute the focal loss.

        Parameters
        ----------
        y_pred : PyTorch tensor (dtype=float)
            Predicted real-valued logits (not probabilities).
            The array shape should be consistent with the targets below.
        y_true : PyTorch tensor (dtype=int)
            Ground truth targets with values in {0,1}.
            The array shape should be consistent with the predictions above.
        '''

        # binary cross entropy
        # bce_terms = -(torch.multiply(y_true, torch.log(torch.sigmoid(y_pred))) \
        #               + torch.multiply((1-y_true), torch.log(1-torch.sigmoid_(y_pred))))
        bce_terms = nn.functional.binary_cross_entropy_with_logits(y_pred,
                                                                   y_true.type(y_pred.dtype),
                                                                   reduction='none')

        # weighting factors
        # balance_weights = torch.where(y_true==1, self.pos_weight, 1.).type(y_pred.dtype).to(y_pred.device)
        balance_weights = torch.ones_like(y_pred)
        balance_weights[y_true==1] = self.pos_weight
        focal_weights = (1 - torch.exp(-bce_terms))**self.focal_gamma

        # weighted mean/sum
        loss = self.reduce(balance_weights * focal_weights * bce_terms)

        return loss

