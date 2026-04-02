"""
olsvered — Closed-form OLS without inversion or normalisation.

Reference: "Solving The Ordinary Least Squares in Closed Form, Without
Inversion or Normalization" — Vered Senderovich Madar & Sandra Batista.
https://arxiv.org/abs/2301.01854
"""

from olsvered.olsvered import (  # noqa: F401
    modified_cholesky,
    back_substitute,
    solve_ols,
    simplified_gram_schmidt,
    weighted_generalized_inverse,
    lu_solve_gram,
    lu_solve_gram_vec,
    lu_inverse_gram,
    lu_damped_inverse_f32,
    eigh_f32,
    apply_kfac_eigen_f32,
    eigh_topk_f32,
    apply_kfac_lowrank_f32,
    randomized_eigh_f32,
)

__all__ = [
    "modified_cholesky",
    "back_substitute",
    "solve_ols",
    "simplified_gram_schmidt",
    "weighted_generalized_inverse",
    "lu_solve_gram",
    "lu_solve_gram_vec",
    "lu_inverse_gram",
    "lu_damped_inverse_f32",
]
