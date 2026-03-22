//! Pure-Rust implementations of three closed-form OLS algorithms.
//!
//! Reference: "Solving The Ordinary Least Squares in Closed Form, Without
//! Inversion or Normalization" — Vered Senderovich Madar & Sandra Batista.
//! <https://arxiv.org/abs/2301.01854>
//!
//! All functions operate on `nalgebra::DMatrix<f64>` / `DVector<f64>`.
//! No PyO3 or FFI dependencies — this module is the pure mathematical core.

use nalgebra::{DMatrix, DVector};
use thiserror::Error;

/// Errors returned by olsvered algorithms.
#[derive(Debug, Error, PartialEq)]
pub enum OlsveredError {
    /// Row count of X does not match length of y.
    #[error("Dimension mismatch: X has {x_rows} rows, y has {y_len} elements")]
    DimensionMismatch { x_rows: usize, y_len: usize },

    /// A diagonal entry of the upper-triangular factor is (near) zero,
    /// indicating a singular or near-singular Gram matrix.
    #[error("Zero pivot at diagonal position {index} — matrix may be singular")]
    ZeroPivot { index: usize },

    /// Weight matrix W is not n×n where n = number of rows in X.
    #[error("Weight matrix W must be {n}x{n}, got {rows}x{cols}")]
    WeightDimension { n: usize, rows: usize, cols: usize },

    /// LU solve returned None — matrix is singular.
    #[error("Singular matrix: LU solve failed")]
    SingularMatrix,
}

// ---------------------------------------------------------------------------
// Algorithm 1 — Modified Cholesky (LU-based row normalisation)
// ---------------------------------------------------------------------------

/// Algorithm 1: LU-based Gram matrix decomposition with row normalisation.
///
/// Augments `[X | y]`, computes the Gram matrix `G = [X|y]ᵀ[X|y]`,
/// LU-decomposes G, and returns the **row-normalised** upper triangular
/// factor C whose diagonal entries are all 1.
///
/// # Arguments
/// * `x` — Design matrix of shape `(n, p)`
/// * `y` — Response vector of length `n`
///
/// # Returns
/// `C` — Upper triangular matrix of shape `(p+1, p+1)` with unit diagonal.
/// Pass to [`back_substitute`] to recover OLS coefficients.
///
/// # Errors
/// * [`OlsveredError::DimensionMismatch`] if `x.nrows() != y.len()`
/// * [`OlsveredError::ZeroPivot`] if any diagonal of U is ≈ 0
pub fn modified_cholesky(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<DMatrix<f64>, OlsveredError> {
    let n = x.nrows();
    let p = x.ncols();

    if n != y.len() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: n,
            y_len: y.len(),
        });
    }

    // Augment X with y as the last column: shape (n, p+1)
    let mut xy = DMatrix::zeros(n, p + 1);
    xy.columns_mut(0, p).copy_from(x);
    xy.column_mut(p).copy_from(y);

    // Gram matrix G = Xyᵀ @ Xy, shape (p+1, p+1)
    let gram = xy.transpose() * &xy;

    // LU decomposition (partial pivot)
    let lu = nalgebra::linalg::LU::new(gram);
    let u = lu.u(); // upper triangular factor

    // Check for zero pivots on the X-columns only (indices 0..p).
    // The last pivot (index p, the y-column) may be zero when y is exactly
    // in the column space of X (exact fit with no residuals) — this is valid.
    for i in 0..p {
        if u[(i, i)].abs() < f64::EPSILON * 1e6 {
            return Err(OlsveredError::ZeroPivot { index: i });
        }
    }

    // Row-normalise: C[i, :] = U[i, :] / U[i, i]  so diag(C) = 1.
    // Skip the last row if its pivot is zero (exact-fit case) — that row is
    // never accessed during back-substitution.
    let dim = u.nrows();
    let mut c = u.clone();
    for i in 0..dim {
        let d = c[(i, i)];
        if d.abs() < f64::EPSILON * 1e6 {
            c[(i, i)] = 1.0; // normalise diagonal to 1 even for zero-pivot rows
            continue;
        }
        for j in i..dim {
            c[(i, j)] /= d;
        }
    }

    Ok(c)
}

// ---------------------------------------------------------------------------
// Back-substitution (companion to Algorithm 1)
// ---------------------------------------------------------------------------

/// Recovers OLS beta coefficients from the C matrix produced by
/// [`modified_cholesky`] via back-substitution.
///
/// Sets `betas[p] = -1`, then for `i` from `p-1` downto `0`:
/// `betas[i] = -(C[i, :] · betas)`.  Returns `betas[0..p]`.
///
/// # Arguments
/// * `c` — `(p+1, p+1)` unit-diagonal upper triangular matrix
///
/// # Returns
/// `beta` — OLS coefficient vector of length `p`
pub fn back_substitute(c: &DMatrix<f64>) -> Result<DVector<f64>, OlsveredError> {
    let dim = c.nrows(); // p + 1
    let mut betas = DVector::zeros(dim);
    betas[dim - 1] = -1.0;

    for i in (0..dim - 1).rev() {
        // betas[i] = -(C[i, :] · betas)
        let dot: f64 = (0..dim).map(|j| c[(i, j)] * betas[j]).sum();
        betas[i] = -dot;
    }

    // Return first p elements (drop the auxiliary -1 entry)
    Ok(DVector::from_iterator(
        dim - 1,
        (0..dim - 1).map(|i| betas[i]),
    ))
}

// ---------------------------------------------------------------------------
// Combined solver
// ---------------------------------------------------------------------------

/// Full OLS solver: equivalent to [`modified_cholesky`] + [`back_substitute`].
///
/// # Arguments
/// * `x` — Design matrix `(n, p)`
/// * `y` — Response vector `(n,)`
///
/// # Returns
/// `beta` — OLS coefficient vector `(p,)`
pub fn solve_ols(
    x: &DMatrix<f64>,
    y: &DVector<f64>,
) -> Result<DVector<f64>, OlsveredError> {
    let c = modified_cholesky(x, y)?;
    back_substitute(&c)
}

// ---------------------------------------------------------------------------
// Algorithm 2 — Simplified Gram-Schmidt Orthogonalisation (SGSO)
// ---------------------------------------------------------------------------

/// Algorithm 2: Non-normalised Gram-Schmidt orthogonalisation (SGSO).
///
/// Produces an orthogonal (but **not** orthonormal) basis Q for the column
/// space of X.  Avoids all square-root computations — only dot products.
///
/// For each column `j`:
/// ```text
/// q_j = x_j  -  Σ_{i<j}  (x_j · q_i) / (q_i · q_i)  *  q_i
/// ```
///
/// # Arguments
/// * `x` — Input matrix `(n, p)`
///
/// # Returns
/// `Q` — Matrix `(n, p)` with mutually orthogonal (un-normalised) columns
pub fn simplified_gram_schmidt(x: &DMatrix<f64>) -> Result<DMatrix<f64>, OlsveredError> {
    let n = x.nrows();
    let p = x.ncols();
    let mut q = DMatrix::zeros(n, p);

    for j in 0..p {
        let mut qj = x.column(j).clone_owned();

        for i in 0..j {
            let qi = q.column(i);
            let num = qj.dot(&qi);
            let den = qi.dot(&qi);
            // Skip numerically zero columns to avoid NaN
            if den.abs() > f64::EPSILON * 1e6 {
                qj -= (num / den) * qi;
            }
        }

        q.column_mut(j).copy_from(&qj);
    }

    Ok(q)
}

// ---------------------------------------------------------------------------
// Algorithm 3 — Weighted Generalised Inverse
// ---------------------------------------------------------------------------

/// Algorithm 3: Weighted generalised inverse `(XᵀWX)⁻¹ Xᵀ W`.
///
/// Computes the weighted least-squares coefficient matrix without explicitly
/// inverting `XᵀWX`. Uses LU solve internally.
///
/// # Arguments
/// * `x` — Design matrix `(n, p)`
/// * `w` — Positive-definite weight matrix `(n, n)` (e.g. a kinship matrix)
///
/// # Returns
/// `G` — Weighted generalised inverse of shape `(p, n)`.
///       For weighted OLS: `beta = G @ y`.
///
/// # Errors
/// * [`OlsveredError::WeightDimension`] if W is not `n×n`
/// * [`OlsveredError::SingularMatrix`] if `XᵀWX` is singular
pub fn weighted_generalized_inverse(
    x: &DMatrix<f64>,
    w: &DMatrix<f64>,
) -> Result<DMatrix<f64>, OlsveredError> {
    let n = x.nrows();

    if w.nrows() != n || w.ncols() != n {
        return Err(OlsveredError::WeightDimension {
            n,
            rows: w.nrows(),
            cols: w.ncols(),
        });
    }

    // xᵀ W  — shape (p, n)
    let xtw = x.transpose() * w;

    // XᵀWX  — shape (p, p)
    let xtwx = &xtw * x;

    // Solve XᵀWX · G = Xᵀ W  for G, shape (p, n)
    // Equivalent to G = (XᵀWX)⁻¹ Xᵀ W without explicit inversion
    let lu = nalgebra::linalg::LU::new(xtwx);
    let result = lu.solve(&xtw).ok_or(OlsveredError::SingularMatrix)?;

    Ok(result)
}
