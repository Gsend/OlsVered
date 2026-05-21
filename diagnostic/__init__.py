"""
OlsVered Diagnostic Package
===========================

Tools for measuring inversion fidelity in layer-wise neural network back-propagation,
comparing naive pseudo-inverse against K-FAC-A regularized (Gaussian-posterior)
inversion.

The diagnostic answers: when we back-propagate a target through a single network
layer, how much does the recovered input distribution drift from the true forward
activation distribution? And does using the K-FAC A factor (empirical activation
covariance) as a prior reduce the drift?

Modules
-------
vered_solve
    Inversion-free SPD solver using Cholesky factorization + cholesky_solve
    with progressive damping fallback. Mirrors OlsSMKFAC._decompose_lu pattern.
inversion
    Two inversion methods: naive pseudo-inverse (Method N) and K-FAC-A
    Gaussian-posterior conditional mean (Method K).
"""

from diagnostic import vered_solve, inversion

__all__ = ["vered_solve", "inversion"]
