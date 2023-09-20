import math
from typing import Tuple, Union

import lightning.pytorch as pl
import torch
import torchvision as tv
import torchvision.transforms as T
from torch.utils.data import DataLoader, Dataset, TensorDataset
from torchmetrics import Accuracy

from models import LogSoftmaxModule

cifar10_normalize_values = {
    'mean': [0.4914, 0.4822, 0.4465],
    'std': [0.2470, 0.2435, 0.2616],
}


def get_transforms(train=True, require_normalize=False) -> T.Compose:
    state = 'train' if train else 'val'
    data_transforms = {
        'train': [
            T.AutoAugment(policy=T.AutoAugmentPolicy.CIFAR10),
            T.RandomHorizontalFlip(p=0.5),
            T.ToTensor(),
        ],
        'val': [
            T.ToTensor(),
        ],
    }
    transform_list = data_transforms[state]
    if require_normalize:
        transform_list = data_transforms[state] + \
            [T.Normalize(
                mean=cifar10_normalize_values['mean'],
                std=cifar10_normalize_values['std'])
             ]

    return T.Compose(transform_list)


def load_cifar10(train=True, require_normalize=False) -> Dataset:
    """Return CIFAR10 dataset."""
    dataset = tv.datasets.CIFAR10(
        './data', download=True, train=train, transform=get_transforms(train, require_normalize))
    return dataset


def inverse_normalize(batch: torch.Tensor, normalize_values: dict) -> torch.Tensor:
    """Convert a tensor to their original scale."""
    device = batch.get_device()
    mean = torch.Tensor(normalize_values['mean']).to(device)
    std = torch.Tensor(normalize_values['std']).to(device)
    return batch * std.view(1, -1, 1, 1) + mean.view(1, -1, 1, 1)


def get_labels(dataloader: DataLoader) -> torch.Tensor:
    """Extract labels from a dataloader."""
    n_samples = len(dataloader.dataset)
    labels = torch.zeros(n_samples, dtype=torch.long)
    start = 0
    for batch in dataloader:
        _, y = batch
        n = len(y)
        end = start + n
        labels[start: end] = y
        start = end
    return labels


def get_data(dataloader: DataLoader) -> torch.Tensor:
    """Extract data from a dataloader."""
    X = []
    for batch in dataloader:
        x, _ = batch
        X.append(x)
    return torch.concat(X)


def evaluate_dataloader(model: Union[pl.LightningModule, torch.nn.Module],
                        dataloader: DataLoader,
                        num_classes=10) -> float:
    """Compute accuracy."""
    if not isinstance(model, pl.LightningModule):
        model = LogSoftmaxModule(model)

    trainer = pl.Trainer(enable_progress_bar=False, logger=False)
    predictions = trainer.predict(model, dataloader)
    predictions = torch.concat(predictions).cpu()
    y = get_labels(dataloader)
    metric = Accuracy(task="multiclass", num_classes=num_classes)
    acc = metric(predictions, y)
    return acc.item()


def choose_dataset(dataset: Dataset, n_sample: Union[int, float], num_workers=1) -> Dataset:
    """Random choose n samples from a dataset without replacement."""
    assert (isinstance(n_sample, int) or n_sample <
            1) and n_sample > 0, 'n_sample is invalid.'
    assert n_sample < len(dataset), 'This function does not allow replacement.'
    if isinstance(n_sample, float):
        n_sample = math.floor(len(dataset) * n_sample)
    dataloader = DataLoader(dataset, batch_size=512,
                            shuffle=True, num_workers=num_workers)
    X = []
    Y = []
    n = 0
    for batch in dataloader:
        x, y = batch
        n += len(x)
        X.append(x)
        Y.append(y)
        if n >= n_sample:
            break
    X = torch.concat(X)[:n_sample]
    Y = torch.concat(Y)[:n_sample]
    return TensorDataset(X, Y)


def find_clip_range(dataset: Dataset) -> Tuple[float, float]:
    """Return the range of a dataset.
    WARNING: Adversarial examples should NOT use a clip range after normalization.
    The scale of the perturbation will be wrong.
    """
    max_x = -torch.inf
    min_x = torch.inf
    for x, _ in dataset:
        _max = x.max()
        _min = x.min()
        if max_x < _max:
            max_x = _max
        if min_x > _min:
            min_x = _min
    return min_x.item(), max_x.item()
