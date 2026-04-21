"""olssm_vision — SLAM / AR / SfM primitives on top of olssm.

Top-level public API is re-exported here for convenience.

Submodules:
    diagnostics : degeneracy / gauge detection, leverage reports
    slam        : online covariance-on-demand queries
    ba          : bundle-adjust LM outer loop (wraps the Rust solver)
"""

from . import diagnostics, slam, ba

# Re-export Rust extension entry points if built.
try:
    from olssm_vision._rust import (  # type: ignore  # built by maturin
        build_schur,
        backsubstitute_points,
        whitened_cholesky,
        pivot_profile,
        detect_near_singular,
        sparsity_template,
        covariance_column,
        covariance_diagonal,
        covariance_block,
    )
except ImportError:  # pragma: no cover
    # Rust extension not built yet — pure-Python fallbacks live in each module.
    build_schur = None
    backsubstitute_points = None
    whitened_cholesky = None
    pivot_profile = None
    detect_near_singular = None
    sparsity_template = None
    covariance_column = None
    covariance_diagonal = None
    covariance_block = None

__all__ = [
    "diagnostics",
    "slam",
    "ba",
    "build_schur",
    "backsubstitute_points",
    "whitened_cholesky",
    "pivot_profile",
    "detect_near_singular",
    "sparsity_template",
    "covariance_column",
    "covariance_diagonal",
    "covariance_block",
]

__version__ = "0.1.0"
