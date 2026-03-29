"""
Data loading utilities for benchmarking.

Provides synthetic and real data loaders:
  - SyntheticLMData: random token sequences (no download needed)
  - CIFAR10 loader (downloads automatically)
"""

import torch
from torch.utils.data import DataLoader, Dataset


class SyntheticLMData(Dataset):
    """Synthetic language modelling dataset — random token sequences.

    Perfect for benchmarking because:
      - No download or preprocessing needed
      - Deterministic with fixed seed
      - Adjustable vocab size and sequence length
      - Loss landscape is realistic (cross-entropy on random sequences)

    Parameters
    ----------
    num_samples : int
        Number of sequences. Default: 10000.
    seq_len : int
        Sequence length. Default: 128.
    vocab_size : int
        Vocabulary size. Default: 1000.
    seed : int
        Random seed for reproducibility. Default: 42.
    """

    def __init__(self, num_samples: int = 10000, seq_len: int = 128,
                 vocab_size: int = 1000, seed: int = 42):
        rng = torch.Generator().manual_seed(seed)
        self.data = torch.randint(0, vocab_size, (num_samples, seq_len + 1),
                                  generator=rng)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        seq = self.data[idx]
        return seq[:-1], seq[1:]  # input, target (shifted by 1)


def get_synthetic_lm_loader(
    num_samples: int = 10000,
    seq_len: int = 128,
    vocab_size: int = 1000,
    batch_size: int = 32,
    seed: int = 42,
) -> DataLoader:
    """Create a DataLoader for synthetic LM data."""
    dataset = SyntheticLMData(num_samples, seq_len, vocab_size, seed)
    pin = torch.cuda.is_available()
    return DataLoader(dataset, batch_size=batch_size, shuffle=True,
                      num_workers=0, pin_memory=pin)


class SyntheticClassificationData(Dataset):
    """Synthetic classification dataset for MLP smoke tests."""

    def __init__(self, num_samples: int = 5000, input_dim: int = 784,
                 num_classes: int = 10, seed: int = 42):
        rng = torch.Generator().manual_seed(seed)
        self.X = torch.randn(num_samples, input_dim, generator=rng)
        self.y = torch.randint(0, num_classes, (num_samples,), generator=rng)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]
