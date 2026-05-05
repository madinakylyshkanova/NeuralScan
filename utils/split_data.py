from torch.utils.data import Subset
import random


def create_splits(dataset, seed=42):
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)

    n = len(indices)
    train_end = int(0.8 * n)
    val_end = int(0.9 * n)

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    return (
        Subset(dataset, train_idx),
        Subset(dataset, val_idx),
        Subset(dataset, test_idx),
    )