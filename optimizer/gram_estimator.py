"""
Abstract interface for Gram matrix estimators used by K-FAC optimizers.

Defining a formal interface decouples KFACHooks from the optimizer classes,
enabling alternative implementations (online EVD, mini-batch averaging,
per-layer EMA with different decay rates, etc.) without modifying optimizer code.

Usage
-----
The default implementation is KFACHooks.  To inject a custom estimator::

    class MyEstimator(GramMatrixEstimator):
        ...

    opt = OlsSMKFAC(model, gram_estimator=MyEstimator(model))
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Dict, List, Optional, Tuple

import torch
import torch.nn as nn


class GramMatrixEstimator(ABC):
    """Abstract base class for K-FAC Gram matrix estimators.

    An estimator registers hooks on a model's layers, accumulates
    input-activation (A) and output-gradient (G) statistics per batch,
    and exposes averaged (A, G) pairs on demand.

    Lifecycle::

        estimator.enable()                  # attach hooks
        for step in range(update_freq):
            loss = model(x); loss.backward()  # hooks accumulate
        factors = estimator.get_factors()   # read averages
        estimator.clear()                   # reset for next window
        # ...
        estimator.remove()                  # detach hooks, free memory
    """

    # ------------------------------------------------------------------
    # Abstract methods — must be implemented by every subclass
    # ------------------------------------------------------------------

    @abstractmethod
    def enable(self) -> None:
        """Attach forward and backward hooks to all tracked layers."""

    @abstractmethod
    def get_factors(self) -> Dict[nn.Module, Tuple[torch.Tensor, torch.Tensor]]:
        """Return averaged Gram matrices (A, G) for each tracked layer.

        Returns
        -------
        dict mapping ``nn.Module`` → ``(A, G)`` where:
            A : (d_in,  d_in)  float tensor — input covariance  E[xᵀx]
            G : (d_out, d_out) float tensor — gradient covariance E[δᵀδ]

        Only layers that have received at least one forward+backward pass
        are included.  Call ``clear()`` after reading to reset accumulators.
        """

    @abstractmethod
    def clear(self) -> None:
        """Reset all accumulator tensors.

        Call after ``get_factors()`` to start a fresh accumulation window.
        """

    @abstractmethod
    def remove(self) -> None:
        """Detach all hooks from the model and free cached tensors."""

    @property
    @abstractmethod
    def linear_layers(self) -> List[nn.Module]:
        """Ordered list of all tracked ``nn.Linear`` / ``nn.Conv2d`` layers."""

    # ------------------------------------------------------------------
    # Optional methods — subclasses may override
    # ------------------------------------------------------------------

    def n_samples_accumulated(self, module: Optional[nn.Module] = None) -> int:
        """Return the number of samples accumulated since the last ``clear()``.

        Default implementation returns 0 — override to provide real counts.

        Parameters
        ----------
        module : nn.Module or None
            If None, returns the count for the first tracked layer as a
            proxy for the whole network.
        """
        return 0

    @property
    def is_enabled(self) -> bool:
        """True if hooks are currently attached."""
        return False
