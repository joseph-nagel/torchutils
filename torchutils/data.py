'''
Data processing.

Summary
-------
Auxiliary functions for typical data processing tasks are contained.
Most notably, the function 'mean_std_over_dataset' computes
the mean and std. of a dataset in batch-wise processing.
The transformer class 'GaussianNoise' can be used to add noise to images.
Balancing sampling for imbalanced datasets can be found in 'BalancedSampler'.
In addition, 'image2tensor' and 'tensor2image' establish converters
between Numpy arrays and PyTorch tensors for image data.
They are compliant with the standard shape conventions.

Notes
-----
Notice that plt.imshow requires (no_rows, no_cols, 3/4)-shaped arrays with
ints in [0,255] or floats in [0,1], while torchvision.transforms.ToTensor yields
(no_samples, no_channels, no_rows, no_cols)-sized tensors with float32s in [0,1].

'''

import numpy as np
import torch
from torch.utils.data import DataLoader, Sampler

def mean_std_over_dataset(data_set, batch_size=1, channel_wise=False, verbose=True):
    '''
    Calculate mean and std. in a batch-wise sweep over the dataset.

    Parameters
    ----------
    data_set : PyTorch DataSet object
        Set of data to be analyzed.

    Returns
    -------
    mean : float or array
        Mean value or channel-wise mean values.
    std : float or array
        Standard deviation or channel-wise standard deviations.

    '''

    # data loader
    data_loader = DataLoader(data_set, batch_size=batch_size, shuffle=False)

    # mean and std.
    if not channel_wise:
        # mean
        mean = 0.
        for images, labels in data_loader:
            # print('{}'.format(images.numpy().shape))
            mean += np.mean(images.numpy())
        mean /= len(data_loader)
        # std.
        std = 0.
        for images, labels in data_loader:
            std += np.mean((images.numpy() - mean)**2)
        std /= len(data_loader) - 1
        std = np.sqrt(std)

    # channel-wise mean and std.
    else:
        # mean
        no_summands = 0.
        mean = np.zeros(3)
        for images, labels in data_loader:
            mean += np.sum(images.numpy(), axis=(0,2,3))
            no_summands += np.size(images.numpy()[:,0,:,:])
        mean /= no_summands
        # std.
        no_summands = 0.
        std = np.zeros(3)
        for images, labels in data_loader:
            std += np.sum((images.numpy() - mean.reshape(1,-1,1,1))**2, axis=(0,2,3))
            no_summands += np.size(images.numpy()[:,0,:,:])
        std /= no_summands - 1
        std = np.sqrt(std)

    if verbose:
        print('Mean: {}'.format(np.array2string(np.array(mean), precision=4)))
        print('Std.: {}'.format(np.array2string(np.array(std), precision=4)))

    return mean, std

class GaussianNoise(object):
    '''
    Gaussian noise corruptions.

    Summary
    -------
    The class realizes a transformer corrupting images with Gaussian noise.
    This can be used for data augmentation or robustness evaluation.

    '''

    def __init__(self, noise_std=1.0):
        self.noise_std = noise_std

    def __call__(self, X):
        X_noisy = X + torch.randn_like(X) * self.noise_std
        return X_noisy

class BalancedSampler(Sampler):
    '''
    Balanced sampling of imbalanced datasets.

    Summary
    -------
    In order to deal with an imbalanced classification dataset,
    an appropriate over/undersampling scheme is implemented.
    Here, samples are taken with replacement from the set, such that
    all classes are equally likely to occur in the training mini-batches.
    This might be especially helpful in combination with data augmentation.
    Different weights for samples in the empirical loss would be an alternative.

    Parameters
    ----------
    data_set : PyTorch dataset
        Imbalanced dataset to be over/undersampled.
    no_samples : int or None
        Number of samples to draw in one epoch.
    indices : array_like or None
        Subset of indices that are sampled.

    '''

    def __init__(self, dataset, no_samples=None, indices=None):
        self.indices = list(range(len(dataset)))
        if no_samples is None:
            self.no_samples = len(dataset) if indices is None else len(indices)
        else:
            self.no_samples = no_samples

        # class occurrence counts
        data_loader = DataLoader(dataset, batch_size=1, shuffle=False)
        labels_list = []
        for image, label in data_loader:
            labels_list.append(label)
        labels_tensor = torch.cat(labels_list, dim=0)
        unique_labels, counts = torch.unique(labels_tensor, return_counts=True)

        # unnormalized probabilities
        weights_for_class = 1.0 / counts.float()
        weights_for_index = torch.tensor(
            [weights_for_class[labels_tensor[idx]] for idx in self.indices]
        )

        # zero indices
        if indices is not None:
            zero_ids = np.setdiff1d(self.indices, indices).tolist()
            weights_for_index[zero_ids] = torch.tensor(0.0)

        # balanced sampling distribution
        self.categorical = torch.distributions.Categorical(probs=weights_for_index)

    def __iter__(self):
        return (idx for idx in self.categorical.sample((self.no_samples,)))

    def __len__(self):
        return self.no_samples

def image2tensor(image, unsqueeze=True):
    '''
    Convert image array to PyTorch tensor.

    Parameters
    ----------
    image : array
    unsqueeze : bool

    Returns
    -------
    tensor : PyTorch tensor

    '''

    if image.ndim == 2: # (no_rows, no_cols)
        tensor = torch.from_numpy(image)
    elif image.ndim == 3: # (no_rows, no_cols, no_channels)
        tensor = torch.from_numpy(image.transpose(2, 0, 1))
    elif image.ndim == 4: # (no_samples, no_rows, no_channels)
        tensor = torch.from_numpy(image.transpose(0, 3, 1, 2))

    if unsqueeze:
        for _ in range(4 - image.ndim):
            tensor = tensor.unsqueeze(0)

    return tensor # (no_samples, no_channels, no_rows, no_colums)

def tensor2image(tensor, squeeze=True):
    '''
    Convert PyTorch tensor to image array.

    Parameters
    ----------
    tensor : PyTorch tensor
    squeeze : bool

    Returns
    -------
    image : array

    '''

    if tensor.ndim == 2: # (no_rows, no_cols)
        image = tensor.numpy()
    elif tensor.ndim == 3: # (no_channels, no_rows, no_cols)
        image = tensor.numpy().transpose((1, 2, 0))
    elif tensor.ndim == 4: # (no_samples, no_channels, no_rows, no_cols)
        image = tensor.numpy().transpose((0, 2, 3, 1))

    if squeeze:
        image = image.squeeze()

    return image # (no_samples, no_rows, no_cols, no_channels)

