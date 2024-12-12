'''
Analysis tools.

Summary
-------
This module collects tools for analyzing trained models.
At the moment, this only includes the function 'confusion_matrix'
to calculate and compile the confusion matrix of a classifier.

'''

import torch


def confusion_matrix(
    classifier,
    data_loader,
    num_epochs=1,
    num_classes=None,
    **kwargs
):
    '''
    Calculate the confusion matrix for a classifier and data loader.

    Summary
    -------
    Given a classifier and a data loader, the confusion matrix is calculated.
    The following convention is adopted for the entries of the matrix.
    While the rows relate to the ground-truth targets,
    the columns refer to the model predictions.

    '''

    classifier.train(False)

    y_pred_list = []
    y_true_list = []

    for _ in range(num_epochs):
        for X_batch, y_batch in data_loader:
            X_batch = X_batch.to(classifier.device)

            with torch.no_grad():
                y_class, y_prob = classifier.predict_top(X_batch, **kwargs)

            y_pred_list.append(y_class.cpu())
            y_true_list.append(y_batch)

    y_pred = torch.cat(y_pred_list, dim=0).squeeze()
    y_true = torch.cat(y_true_list, dim=0)

    confmat = conf_mat(y_true, y_pred, num_classes).numpy()

    return confmat


def conf_mat(y_true, y_pred, num_classes=None):
    '''
    Construct the confusion matrix from predictions and targets.

    Summary
    -------
    The confusion matrix is compiled for integer-encoded
    ground-truth labels and corresponding class predictions.
    This is analogous to sklearn.metrics.confusion_matrix.

    '''

    y_true = y_true.int()
    y_pred = y_pred.int()

    if num_classes is None:
        num_classes = 1 + torch.max(torch.cat((y_true, y_pred), dim=0)).item()

    confmat = torch.zeros(num_classes, num_classes, dtype=torch.int64)
    for r, c in zip(y_true, y_pred):
        confmat[r, c] += 1

    return confmat

