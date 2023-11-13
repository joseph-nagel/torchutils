'''Tests for the classification module.'''

import pytest
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import train_test_split

from torchutils.classification import Classification


@pytest.fixture(params=[1, 10])
def data_num_features(request):
    return request.param


@pytest.fixture(params=[2, 10])
def data_num_classes(request):
    return request.param


@pytest.fixture(params=[1000])
def data_num_samples(request):
    return request.param


@pytest.fixture
def data_classification(data_num_features,
                        data_num_classes,
                        data_num_samples):
    '''Create classification problem.'''

    torch.manual_seed(0)

    num_features = data_num_features
    num_classes = data_num_classes
    num_samples = data_num_samples

    num_outputs = 1 if num_classes==2 else num_classes

    X = torch.randn(num_samples, num_features)
    y = torch.randint(num_classes, size=(num_samples,))

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=0
    )

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_set, batch_size=32, shuffle=False)

    model = nn.Linear(num_features, num_outputs)

    if num_outputs == 1:
        criterion = nn.BCEWithLogitsLoss(reduction='mean')
    else:
        criterion = nn.CrossEntropyLoss(reduction='mean')

    optimizer = torch.optim.Adam(params=model.parameters())

    classifier = Classification(
        model,
        criterion,
        optimizer,
        train_loader,
        test_loader,
        torch.device('cpu')
    )

    return classifier


def test_classifier_inference(data_classification):
    '''Test classifier at inference.'''

    classifier = data_classification
    classifier.train(False)

    X_batch, y_batch = next(iter(classifier.train_loader))
    X_batch = X_batch.to(classifier.device)
    y_batch = y_batch.to(classifier.device)

    with torch.no_grad():
        y_pred = classifier.predict(X_batch)
        y_proba = classifier.predict_proba(X_batch)
        y_topclass, _ = classifier.predict_top(X_batch)

    y_pred = y_pred.cpu()
    y_proba = y_proba.cpu()
    y_topclass = y_topclass.cpu()

    assert y_pred.shape == (X_batch.shape[0], classifier.model.out_features)
    assert y_proba.shape == (X_batch.shape[0], classifier.model.out_features)
    assert y_topclass.numel() == y_batch.numel()

