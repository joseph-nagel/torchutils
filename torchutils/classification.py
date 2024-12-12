'''
Classification problems.

Summary
-------
The core class 'Classification' is provided below.
It mainly establishes a lightweight wrapper for PyTorch classifiers
and equips them with a training loop and testing functionality.

Notes
-----
Regarding PyTorch's classification losses, it is remarked that nn.CrossEntropyLoss
expects (num_samples, num_classes>1) predictions and (num_samples,) targets, while
nn.BCEWithLogitsLoss requires equally shaped predictions and targets with (num_samples, *).
This is realized through corresponding if-constructs in the code.

'''

import torch
import torch.nn as nn

from .tools import moving_average
from .loss import HingeLoss


class Classification(object):
    '''
    Classification model wrapper class.

    Summary
    -------
    This class facilitates the training and testing of PyTorch classifiers.
    It features methods performing a whole training loop and a single epoch.
    Testing on a full data loader is also implemented as a method.
    The most common PyTorch loss functions nn.BCEWithLogitsLoss,
    nn.CrossEntropyLoss and nn.NLLLoss are supported at the moment.
    The implementation aims at being device-agnostic,
    i.e. a GPU is used whenever available, the CPU is used otherwise.

    The class supports both binary and multi-class classification problems.
    For the binary case, it is assumed that the model does not involve a final (log)-sigmoid,
    i.e. the sigmoid is applied to the model output for testing purposes only.
    For the multiclass problem, the model may or may not perform a (log)-softmax operation.
    During testing, only the class with the highest response is compared to ground truth,
    the results of which are not altered under the normalizing softmax function.
    Note that, whenever desired, the logit scores can be easily casted as probabilities.

    Parameters
    ----------
    model : PyTorch module
        Model to be trained.
    criterion : PyTorch loss function
        Loss function criterion.
    optimizer : PyTorch optimizer
        Optimization routine.
    train_loader : PyTorch data loader
        Loader that generates the training data.
    val_loader : PyTorch data loader
        Loader that generates the validation data.
    device : PyTorch device
        Device the computations are performed on.

    '''

    def __init__(
        self,
        model,
        criterion,
        optimizer,
        train_loader,
        val_loader=None,
        device=None
    ):

        # arguments
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.train_loader = train_loader
        self.val_loader = val_loader

        # device
        if device is None:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = device

        self.model = self.model.to(self.device)

    def __call__(self, X):
        '''Call model.'''
        y = self.model(X)
        return y

    def predict(self, X):
        '''Predict outputs.'''
        y = self(X)
        return y

    def predict_proba(self, X):
        '''Predict probabilities.'''
        y = self.predict(X)

        if isinstance(self.criterion, nn.BCEWithLogitsLoss):
            y_probs = torch.sigmoid(y)
        elif isinstance(self.criterion, nn.CrossEntropyLoss):
            y_probs = torch.softmax(y, dim=1)
        elif isinstance(self.criterion, nn.NLLLoss):
            y_probs = torch.exp(y, dim=1)
        elif isinstance(self.criterion, HingeLoss):
            raise NotImplementedError('Platt scaling is not yet implemented')

        return y_probs

    def predict_top(self, X, threshold=0.5):
        '''Predict top class and probability.'''
        y_probs = self.predict_proba(X)

        if isinstance(self.criterion, (nn.BCEWithLogitsLoss, HingeLoss)):
            top_class = (y_probs >= threshold).int()
            top_prob = torch.where(top_class==1, y_probs, 1-y_probs)
        elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
            top_prob, top_class = torch.topk(y_probs, k=1, dim=1)

        return top_class, top_prob

    def train(self, mode=True):
        '''Set training mode of the model.'''
        self.model.train(mode)

    def training(
        self,
        num_epochs,
        log_interval=100,
        threshold=0.5,
        initial_test=True
    ):
        '''Perform a number of training epochs.'''

        self.epoch = 0

        train_losses = []
        val_losses = []
        val_accs = []

        # initial test
        if initial_test:
            train_loss, train_acc = self.test(self.train_loader, threshold=threshold)
            val_loss, val_acc = self.test(self.val_loader, threshold=threshold)

            train_losses.append(train_loss)
            val_losses.append(val_loss)
            val_accs.append(val_acc)

            print('Started training: {}, avg. val. loss: {:.4f}, val. acc.: {:.4f}' \
                  .format(self.epoch, val_loss, val_acc))

        # training loop
        for epoch_idx in range(num_epochs):
            train_loss = self.train_epoch(log_interval)
            train_losses.append(train_loss)
            self.epoch += 1

            if self.val_loader is not None:
                val_loss, val_acc = self.test(threshold=threshold)
                val_losses.append(val_loss)
                val_accs.append(val_acc)

                print('Finished epoch: {}, avg. val. loss: {:.4f}, val. acc.: {:.4f}' \
                      .format(self.epoch, val_loss, val_acc))

        history = {'num_epochs': num_epochs,
                   'train_loss': train_losses,
                   'val_loss': val_losses,
                   'val_acc': val_accs}

        return history

    def train_epoch(self, log_interval=100):
        '''Perform a single training epoch.'''

        self.train(True)

        batch_losses = []
        for batch_idx, (X_batch, y_batch) in enumerate(self.train_loader):

            # device
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            # forward
            y_pred = self.model(X_batch)

            if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                y_batch = y_batch.view(*y_pred.shape).float()
                # y_batch = y_batch.float()
                # y_pred = y_pred.squeeze()
            elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                y_batch = y_batch.view(-1)
            elif isinstance(self.criterion, HingeLoss):
                y_batch[y_batch == 0] = -1
                y_batch = y_batch.float()

            loss = self.criterion(y_pred, y_batch)

            # backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            # analysis
            batch_loss = loss.data.item()
            batch_losses.append(batch_loss)

            if len(batch_losses) < 3:
                running_loss = batch_loss
            else:
                running_loss = moving_average(batch_losses, window=3, mode='last')

            if log_interval is not None:
                if (batch_idx+1) % log_interval == 0 or (batch_idx+1) == len(self.train_loader):
                    print('Epoch: {} ({}/{}), batch loss: {:.4f}, running loss: {:.4f}' \
                          .format(self.epoch+1, batch_idx+1, len(self.train_loader), batch_loss, running_loss))

        return running_loss

    @torch.no_grad()
    def test(
        self,
        test_loader=None,
        num_epochs=1,
        threshold=0.5
    ):
        '''Compute average test loss and accuracy.'''

        self.train(False)

        if test_loader is None:
            test_loader = self.val_loader

        num_total = 0
        num_correct = 0
        test_loss = 0.0

        for epoch_idx in range(num_epochs):
            for X_batch, y_batch in test_loader:

                # device
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                # forward
                y_pred = self.model(X_batch)

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    y_batch = y_batch.view(*y_pred.shape).float()
                    # y_batch = y_batch.float()
                    # y_pred = y_pred.squeeze()
                elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    y_batch = y_batch.view(-1)
                elif isinstance(self.criterion, HingeLoss):
                    y_batch[y_batch == 0] = -1
                    y_batch = y_batch.float()

                loss = self.criterion(y_pred, y_batch)

                # analysis
                test_loss += loss.data.item()
                num_total += X_batch.shape[0]

                if isinstance(self.criterion, nn.BCEWithLogitsLoss):
                    is_correct = (torch.sigmoid(y_pred) >= threshold).squeeze().int() == y_batch.squeeze().int()
                elif isinstance(self.criterion, (nn.CrossEntropyLoss, nn.NLLLoss)):
                    is_correct = torch.max(y_pred, dim=1)[1].squeeze().int() == y_batch.squeeze().int()
                elif isinstance(self.criterion, HingeLoss):
                    is_correct = torch.sign(y_pred).squeeze().int() == y_batch.squeeze().int()

                num_correct += torch.sum(is_correct).item()

        test_acc = num_correct / num_total

        if self.criterion.reduction == 'sum': # averaging over all data
            test_loss /= len(test_loader.dataset)
        elif self.criterion.reduction == 'mean': # averaging over batches
            test_loss /= len(test_loader)

        return test_loss, test_acc

