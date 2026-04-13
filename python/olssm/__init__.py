"""
olssm — Closed-form OLS without inversion or normalisation.

Inversion or Normalization" —  Senderovich  & Sandra .

"""

from olssm.olssm import (  # noqa: F401
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
