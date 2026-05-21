"""diagnostic.models — model classes for arch-coverage experiments."""
from diagnostic.models.deep_mlp import DeepMLP
from diagnostic.models.lnmlp import LNMLP
from diagnostic.models.mlp_autoencoder import MlpAutoencoder
from diagnostic.models.lenet import LeNet

__all__ = ["DeepMLP", "LNMLP", "MlpAutoencoder", "LeNet"]
