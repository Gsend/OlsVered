//! Rust unit tests for the olssm algorithms.
//!
//! Tests are written FIRST (TDD red phase) and define the contract for each
//! algorithm.  Uses the `approx` crate for floating-point comparisons.

use approx::assert_abs_diff_eq;
use nalgebra::{DMatrix, DVector};
use olssm::algorithms::*;

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Build a simple overdetermined 4×2 system with known solution β = [2, 3].
/// X @ [2, 3] = y exactly (no noise).
fn make_4x2_system() -> (DMatrix<f64>, DVector<f64>) {
    // Rows: (1,0), (0,1), (1,1), (2,1)  →  y = 2*x1 + 3*x2
    let x = DMatrix::from_row_slice(
        4,
        2,
        &[1.0, 0.0, 0.0, 1.0, 1.0, 1.0, 2.0, 1.0],
    );
    let y = DVector::from_column_slice(&[2.0, 3.0, 5.0, 7.0]);
    (x, y)
}

/// Build a small random-ish full-rank 5×3 system (fixed seed for determinism).
fn make_5x3_system() -> (DMatrix<f64>, DVector<f64>) {
    let x = DMatrix::from_row_slice(
        5,
        3,
        &[
            1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.1, 0.0, 1.0, 2.0, 3.0, 1.0,
            0.5,
        ],
    );
    let y = DVector::from_column_slice(&[1.0, 2.0, 3.0, 4.0, 5.0]);
    (x, y)
}

// ---------------------------------------------------------------------------
// Algorithm 1 — modified_cholesky
// ---------------------------------------------------------------------------

#[test]
fn test_modified_cholesky_unit_diagonal() {
    let (x, y) = make_4x2_system();
    let c = modified_cholesky(&x, &y).expect("modified_cholesky should succeed");
    for i in 0..c.nrows() {
        assert_abs_diff_eq!(c[(i, i)], 1.0, epsilon = 1e-12);
    }
}

#[test]
fn test_modified_cholesky_upper_triangular() {
    let (x, y) = make_4x2_system();
    let c = modified_cholesky(&x, &y).expect("modified_cholesky should succeed");
    for i in 0..c.nrows() {
        for j in 0..i {
            assert_abs_diff_eq!(c[(i, j)], 0.0, epsilon = 1e-12);
        }
    }
}

#[test]
fn test_modified_cholesky_output_shape() {
    let (x, y) = make_4x2_system();
    let c = modified_cholesky(&x, &y).expect("modified_cholesky should succeed");
    let p = x.ncols();
    assert_eq!(c.nrows(), p + 1);
    assert_eq!(c.ncols(), p + 1);
}

// ---------------------------------------------------------------------------
// Back-substitution
// ---------------------------------------------------------------------------

#[test]
fn test_back_substitute_output_length() {
    let (x, y) = make_4x2_system();
    let c = modified_cholesky(&x, &y).unwrap();
    let beta = back_substitute(&c).unwrap();
    assert_eq!(beta.len(), x.ncols());
}

// ---------------------------------------------------------------------------
// solve_ols — combined solver
// ---------------------------------------------------------------------------

#[test]
fn test_solve_ols_exact_system() {
    // Exact system: β = [2, 3] exactly satisfies X @ β = y
    let (x, y) = make_4x2_system();
    let beta = solve_ols(&x, &y).expect("solve_ols should succeed");
    assert_abs_diff_eq!(beta[0], 2.0, epsilon = 1e-8);
    assert_abs_diff_eq!(beta[1], 3.0, epsilon = 1e-8);
}

#[test]
fn test_solve_ols_overdetermined_residual() {
    // Noisy overdetermined system — residuals should be small
    let x = DMatrix::from_row_slice(
        6,
        2,
        &[1.0, 1.0, 1.0, 2.0, 1.0, 3.0, 1.0, 4.0, 1.0, 5.0, 1.0, 6.0],
    );
    // y ≈ 1 + 2*x2 with small noise
    let y = DVector::from_column_slice(&[3.1, 4.9, 7.0, 9.1, 10.9, 13.0]);
    let beta = solve_ols(&x, &y).expect("solve_ols should succeed");
    // Reconstructed y
    let y_hat = &x * &beta;
    let residual_max = (&y_hat - &y).abs().max();
    assert!(residual_max < 0.5, "max residual {residual_max} too large");
}

#[test]
fn test_solve_ols_5x3_system() {
    let (x, y) = make_5x3_system();
    let beta = solve_ols(&x, &y).expect("solve_ols should succeed");
    assert_eq!(beta.len(), 3);
    // Verify residuals are reasonable (5x3 system is near-rank-deficient, so tolerance is wider)
    let residual_max = (&x * &beta - &y).abs().max();
    assert!(residual_max < 3.0, "max residual {residual_max} too large");
}

// ---------------------------------------------------------------------------
// Error handling
// ---------------------------------------------------------------------------

#[test]
fn test_dimension_mismatch_error() {
    let x = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let y = DVector::from_column_slice(&[1.0, 2.0]); // wrong length: 2 ≠ 3
    let result = solve_ols(&x, &y);
    assert!(
        matches!(result, Err(OlsSMError::DimensionMismatch { .. })),
        "expected DimensionMismatch, got {result:?}"
    );
}

// ---------------------------------------------------------------------------
// Algorithm 2 — simplified_gram_schmidt
// ---------------------------------------------------------------------------

#[test]
fn test_gram_schmidt_output_shape() {
    let (x, _) = make_5x3_system();
    let q = simplified_gram_schmidt(&x).expect("SGSO should succeed");
    assert_eq!(q.nrows(), x.nrows());
    assert_eq!(q.ncols(), x.ncols());
}

#[test]
fn test_gram_schmidt_orthogonality() {
    let (x, _) = make_5x3_system();
    let q = simplified_gram_schmidt(&x).expect("SGSO should succeed");
    let qtq = q.transpose() * &q;
    // Off-diagonal entries of QᵀQ must be ≈ 0
    for i in 0..qtq.nrows() {
        for j in 0..qtq.ncols() {
            if i != j {
                assert_abs_diff_eq!(qtq[(i, j)], 0.0, epsilon = 1e-10);
            }
        }
    }
}

#[test]
fn test_gram_schmidt_4x2() {
    let (x, _) = make_4x2_system();
    let q = simplified_gram_schmidt(&x).expect("SGSO should succeed");
    let qtq = q.transpose() * &q;
    assert_abs_diff_eq!(qtq[(0, 1)], 0.0, epsilon = 1e-12);
    assert_abs_diff_eq!(qtq[(1, 0)], 0.0, epsilon = 1e-12);
}

// ---------------------------------------------------------------------------
// Algorithm 3 — weighted_generalized_inverse
// ---------------------------------------------------------------------------

#[test]
fn test_weighted_generalized_inverse_identity_weight() {
    // W = I  →  result = (XᵀX)⁻¹ Xᵀ  (standard Moore-Penrose pseudoinverse)
    let (x, _) = make_5x3_system();
    let n = x.nrows();
    let w = DMatrix::identity(n, n);
    let g = weighted_generalized_inverse(&x, &w)
        .expect("weighted_generalized_inverse should succeed");
    assert_eq!(g.nrows(), x.ncols());
    assert_eq!(g.ncols(), n);
    // Left-inverse property in W-metric: (XᵀX)⁻¹ Xᵀ @ X = I_p
    let should_be_identity = &g * &x;
    let identity = DMatrix::identity(x.ncols(), x.ncols());
    for i in 0..identity.nrows() {
        for j in 0..identity.ncols() {
            assert_abs_diff_eq!(
                should_be_identity[(i, j)],
                identity[(i, j)],
                epsilon = 1e-8
            );
        }
    }
}

#[test]
fn test_weighted_generalized_inverse_dimension_error() {
    let (x, _) = make_4x2_system();
    let w_wrong = DMatrix::identity(3, 3); // wrong size: n=4 but W is 3×3
    let result = weighted_generalized_inverse(&x, &w_wrong);
    assert!(
        matches!(result, Err(OlsSMError::WeightDimension { .. })),
        "expected WeightDimension error"
    );
}

#[test]
fn test_weighted_generalized_inverse_output_shape() {
    let (x, _) = make_5x3_system();
    let n = x.nrows();
    let w = DMatrix::identity(n, n);
    let g = weighted_generalized_inverse(&x, &w).unwrap();
    assert_eq!(g.nrows(), x.ncols()); // p rows
    assert_eq!(g.ncols(), n); // n cols
}

// ---------------------------------------------------------------------------
// Algorithm 4 — lu_solve_gram
// ---------------------------------------------------------------------------

#[test]
fn test_lu_solve_gram_identity() {
    // gram = I → solution = rhs
    let p = 4;
    let k = 2;
    let gram = DMatrix::identity(p, p);
    let rhs = DMatrix::from_row_slice(p, k, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
    ]);
    let result = lu_solve_gram(&gram, &rhs).expect("lu_solve_gram should succeed");
    assert_eq!(result.nrows(), p);
    assert_eq!(result.ncols(), k);
    for i in 0..p {
        for j in 0..k {
            assert_abs_diff_eq!(result[(i, j)], rhs[(i, j)], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_lu_solve_gram_matches_solve_ols() {
    // lu_solve_gram(XᵀX, Xᵀy) should match solve_ols(X, y)
    let (x, y) = make_4x2_system();
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    // Convert xty vector to single-column matrix for lu_solve_gram
    let xty_mat = DMatrix::from_column_slice(xty.len(), 1, xty.as_slice());
    let result = lu_solve_gram(&xtx, &xty_mat).expect("lu_solve_gram should succeed");
    let beta_ols = solve_ols(&x, &y).expect("solve_ols should succeed");
    for i in 0..beta_ols.len() {
        assert_abs_diff_eq!(result[(i, 0)], beta_ols[i], epsilon = 1e-10);
    }
}

#[test]
fn test_lu_solve_gram_vec_matches_solve_ols() {
    let (x, y) = make_4x2_system();
    let xtx = x.transpose() * &x;
    let xty = x.transpose() * &y;
    let result = lu_solve_gram_vec(&xtx, &xty).expect("lu_solve_gram_vec should succeed");
    let beta_ols = solve_ols(&x, &y).expect("solve_ols should succeed");
    for i in 0..beta_ols.len() {
        assert_abs_diff_eq!(result[i], beta_ols[i], epsilon = 1e-10);
    }
}

#[test]
fn test_lu_solve_gram_dimension_mismatch() {
    let gram = DMatrix::identity(3, 3);
    let rhs = DMatrix::from_row_slice(4, 1, &[1.0, 2.0, 3.0, 4.0]);
    let result = lu_solve_gram(&gram, &rhs);
    assert!(
        matches!(result, Err(OlsSMError::DimensionMismatch { .. })),
        "expected DimensionMismatch"
    );
}

#[test]
fn test_lu_solve_gram_non_square_error() {
    let gram = DMatrix::from_row_slice(3, 2, &[1.0, 2.0, 3.0, 4.0, 5.0, 6.0]);
    let rhs = DMatrix::from_row_slice(3, 1, &[1.0, 2.0, 3.0]);
    let result = lu_solve_gram(&gram, &rhs);
    assert!(
        matches!(result, Err(OlsSMError::DimensionMismatch { .. })),
        "expected DimensionMismatch for non-square gram"
    );
}

// ---------------------------------------------------------------------------
// lu_inverse_gram
// ---------------------------------------------------------------------------

#[test]
fn test_lu_inverse_gram_identity() {
    let p = 4;
    let gram = DMatrix::identity(p, p);
    let inv = lu_inverse_gram(&gram).expect("lu_inverse_gram should succeed");
    let identity = DMatrix::identity(p, p);
    for i in 0..p {
        for j in 0..p {
            assert_abs_diff_eq!(inv[(i, j)], identity[(i, j)], epsilon = 1e-12);
        }
    }
}

#[test]
fn test_lu_inverse_gram_roundtrip() {
    // gram @ inv(gram) ≈ I
    let (x, _) = make_5x3_system();
    let gram = x.transpose() * &x; // shape (3, 3)
    let inv = lu_inverse_gram(&gram).expect("lu_inverse_gram should succeed");
    let product = &gram * &inv;
    let identity = DMatrix::identity(gram.nrows(), gram.ncols());
    for i in 0..identity.nrows() {
        for j in 0..identity.ncols() {
            assert_abs_diff_eq!(product[(i, j)], identity[(i, j)], epsilon = 1e-8);
        }
    }
}

#[test]
fn test_lu_inverse_gram_output_shape() {
    let p = 5;
    let gram = DMatrix::identity(p, p);
    let inv = lu_inverse_gram(&gram).unwrap();
    assert_eq!(inv.nrows(), p);
    assert_eq!(inv.ncols(), p);
}
