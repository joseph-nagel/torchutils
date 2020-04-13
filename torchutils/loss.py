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
Though, as one can see below, the explicit handling of the regularization
in the loss function would require to pass the relevant model parameters.

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

    def __init__(self, params=None, c=1.0, squared=False, reduction='mean'):
        super().__init__()
        self.params = params
        self.c = c
        self.squared = squared
        self.reduction = reduction
        if self.reduction == 'mean':
            self.reduce = torch.mean
        elif self.reduction == 'sum':
            self.reduce = torch.sum

    def forward(self, y_pred, y_true):
        if not self.squared: # hinge loss
            loss = self.reduce(torch.clamp(1 - y_true.squeeze() * y_pred.squeeze(), min=0))
        else: # squared hinge loss
            loss = self.reduce(torch.clamp(1 - y_true.squeeze() * y_pred.squeeze(), min=0) ** 2)
        if self.params is not None and self.c is not None: # l2 penalty
            loss += self.c * self.sum(self.params ** 2)
        return loss

