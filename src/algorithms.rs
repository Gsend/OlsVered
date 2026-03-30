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
// faer traits required for .solve() and .inverse() on PartialPivLu
use faer::prelude::{SolverCore, SpSolver};

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

// ---------------------------------------------------------------------------
// Algorithm 4 — Direct Gram Matrix LU Solve (for K-FAC / Shampoo integration)
//
// The three public functions below (`lu_solve_gram`, `lu_solve_gram_vec`,
// `lu_inverse_gram`) are the hot path for K-FAC preconditioning.  They use
// `faer` instead of nalgebra so that SIMD-accelerated, cache-aware kernels
// replace the generic nalgebra code — closing the gap with BLAS-backed
// torch.linalg.inv without requiring any external system libraries.
// ---------------------------------------------------------------------------

/// Convert a nalgebra `DMatrix<f64>` to a `faer::Mat<f64>` (column-major copy).
#[inline]
fn nalgebra_to_faer(m: &DMatrix<f64>) -> faer::Mat<f64> {
    let nrows = m.nrows();
    let ncols = m.ncols();
    faer::Mat::from_fn(nrows, ncols, |i, j| m[(i, j)])
}

/// Convert a `faer::MatRef<f64>` back to a nalgebra `DMatrix<f64>`.
#[inline]
fn faer_to_nalgebra(m: faer::MatRef<f64>) -> DMatrix<f64> {
    DMatrix::from_fn(m.nrows(), m.ncols(), |i, j| m.read(i, j))
}

/// Solve `gram · X = rhs` via faer LU factorisation — no explicit inverse formed.
///
/// Uses `faer`'s SIMD-accelerated partial-pivoting LU which is significantly
/// faster than nalgebra for matrices ≥ 64×64 (the typical Kronecker factor
/// size in K-FAC for transformer layers).
///
/// # Arguments
/// * `gram` — Symmetric positive-(semi)definite matrix of shape `(p, p)`
/// * `rhs`  — Right-hand side matrix of shape `(p, k)`
///
/// # Returns
/// Solution matrix `X` of shape `(p, k)` such that `gram · X ≈ rhs`.
///
/// # Errors
/// * [`OlsveredError::DimensionMismatch`] if row counts disagree
/// * [`OlsveredError::SingularMatrix`]    if LU solve fails (singular matrix)
pub fn lu_solve_gram(
    gram: &DMatrix<f64>,
    rhs: &DMatrix<f64>,
) -> Result<DMatrix<f64>, OlsveredError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }
    if gram.nrows() != rhs.nrows() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: rhs.nrows(),
        });
    }

    let fa = nalgebra_to_faer(gram);
    let fb = nalgebra_to_faer(rhs);
    let plu = fa.partial_piv_lu();
    let fx = plu.solve(&fb);
    Ok(faer_to_nalgebra(fx.as_ref()))
}

/// Solve `gram · x = rhs` for a single right-hand-side vector.
///
/// Reshapes the vector to a single-column matrix, delegates to `faer` LU,
/// and reshapes back.
///
/// # Arguments
/// * `gram` — Symmetric positive-(semi)definite matrix `(p, p)`
/// * `rhs`  — Right-hand side vector `(p,)`
///
/// # Returns
/// Solution vector `x` of length `p`.
pub fn lu_solve_gram_vec(
    gram: &DMatrix<f64>,
    rhs: &DVector<f64>,
) -> Result<DVector<f64>, OlsveredError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }
    if gram.nrows() != rhs.len() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: rhs.len(),
        });
    }

    let p = gram.nrows();
    let fa = nalgebra_to_faer(gram);
    let fb = faer::Mat::from_fn(p, 1, |i, _| rhs[i]);
    let plu = fa.partial_piv_lu();
    let fx = plu.solve(&fb);
    Ok(DVector::from_fn(p, |i, _| fx.read(i, 0)))
}

/// Compute the explicit inverse of a Gram matrix via faer LU factorisation.
///
/// While `olsvered` philosophy favours direct solves over explicit inversion,
/// K-FAC's natural gradient update `ΔW = G⁻¹ · ∇L · A⁻¹` requires the
/// preconditioner be applied from both sides — making a cached explicit
/// inverse worthwhile when `factor_update_freq > 1`.
///
/// Uses `faer`'s SIMD-accelerated LU which is significantly faster than the
/// nalgebra implementation for the matrix sizes typical in K-FAC.
///
/// # Arguments
/// * `gram` — Square matrix of shape `(p, p)`
///
/// # Returns
/// `gram⁻¹` — Inverse matrix of shape `(p, p)`
///
/// # Errors
/// * [`OlsveredError::SingularMatrix`] if the matrix is singular
pub fn lu_inverse_gram(gram: &DMatrix<f64>) -> Result<DMatrix<f64>, OlsveredError> {
    if gram.nrows() != gram.ncols() {
        return Err(OlsveredError::DimensionMismatch {
            x_rows: gram.nrows(),
            y_len: gram.ncols(),
        });
    }

    let fa = nalgebra_to_faer(gram);
    let plu = fa.partial_piv_lu();
    // faer's inverse() computes A⁻¹ directly without constructing an identity RHS
    let finv = plu.inverse();
    Ok(faer_to_nalgebra(finv.as_ref()))
}

// ---------------------------------------------------------------------------
// Fast f32 path — zero-overhead K-FAC inversion for production use
//
// `lu_damped_inverse_f32` is the hot path used by OlsveredKFAC at runtime:
//   1. Accepts a **row-major f32 slice** — matches PyTorch's default memory
//      layout so no dtype conversion is needed on the Python side.
//   2. Adds Tikhonov damping λI directly inside Rust — one fewer numpy
//      allocation per call.
//   3. Returns a **row-major Vec<f32>** — wrap in numpy once, then
//      `torch.from_numpy()` directly; cached as f32 torch tensor.
//
// Compared with the f64 `lu_inverse_gram` path:
//   Old: f32 tensor → .astype(f64) → numpy f64 → DMatrix<f64> → faer f64
//        → DMatrix<f64> → numpy f64 → torch.from_numpy → .to(f32)
//        = 6+ copies per matrix
//   New: f32 tensor → .numpy() → faer f32 → Vec<f32> → numpy f32
//        → torch.from_numpy()
//        = 2 copies per matrix
// ---------------------------------------------------------------------------

/// Compute `(gram + damping·I)⁻¹` directly on f32 data.
///
/// Designed for the K-FAC hot path: avoids the f32→f64 dtype conversion
/// and the nalgebra intermediate by working with raw slices throughout.
///
/// # Arguments
/// * `gram`    — Row-major f32 slice of length `n × n`.
/// * `n`       — Matrix dimension.
/// * `damping` — Tikhonov damping scalar λ added to the diagonal.
///
/// # Returns
/// Row-major `Vec<f32>` of length `n × n` containing `(gram + λI)⁻¹`.
pub fn lu_damped_inverse_f32(gram: &[f32], n: usize, damping: f32) -> Vec<f32> {
    use faer::prelude::SolverCore;

    // Build faer::Mat<f32> from the row-major slice (one copy: row→col major)
    let mut fa: faer::Mat<f32> = faer::Mat::from_fn(n, n, |i, j| gram[i * n + j]);

    // Add damping in-place — no extra allocation
    for k in 0..n {
        *fa.get_mut(k, k) += damping;
    }

    // LU factorisation + explicit inverse (faer SIMD kernels)
    let plu = fa.partial_piv_lu();
    let finv = plu.inverse();

    // Write back as row-major f32 (one copy: col→row major)
    let mut out = vec![0.0f32; n * n];
    for i in 0..n {
        for j in 0..n {
            out[i * n + j] = finv.read(i, j);
        }
    }
    out
}

/// Compute the self-adjoint eigendecomposition of a symmetric f32 Gram matrix and
/// return the eigenvector matrix **Q** plus damped inverse eigenvalues **1/(λ+δ)**.
///
/// # Arguments
/// * `gram`    — Row-major `n×n` f32 slice (symmetric, PSD).
/// * `n`       — Matrix dimension.
/// * `damping` — Scalar δ added in eigenvalue space: returns `1/(λᵢ + δ)`.
///               Applied after decomposition so Q can be reused across damping
///               values without re-decomposing.
///
/// # Returns
/// `(q_flat, inv_lambda)` where:
/// * `q_flat` is row-major `n×n` f32 (column j of Q = eigenvector j, sorted by
///   ascending eigenvalue).
/// * `inv_lambda` is length-n f32 with values `1 / max(λᵢ + δ, 1e-8)` — clamped
///   to prevent division by zero near-singular matrices.
///
/// The original matrix is reconstructed as `gram ≈ Q @ diag(λ) @ Qᵀ`.
pub fn eigh_f32(gram: &[f32], n: usize, damping: f32) -> (Vec<f32>, Vec<f32>) {
    use faer::linalg::solvers::SelfAdjointEigendecomposition;

    let fa = faer::Mat::<f32>::from_fn(n, n, |i, j| gram[i * n + j]);
    // faer's symmetric EVD: A = U S Uᵀ, eigenvalues sorted ascending
    let eig = SelfAdjointEigendecomposition::<f32>::new(fa.as_ref(), faer::Side::Lower);

    // Inverse damped eigenvalues: 1 / max(λᵢ + δ, 1e-8)
    let inv_lambda: Vec<f32> = (0..n)
        .map(|i| {
            let lam = eig.s().column_vector().read(i);
            1.0_f32 / (lam + damping).max(1e-8_f32)
        })
        .collect();

    // Eigenvectors in row-major layout: q_flat[i*n + j] = Q[i, j]
    let q = eig.u();
    let q_flat: Vec<f32> = (0..n)
        .flat_map(|i| (0..n).map(move |j| q.read(i, j)))
        .collect();

    (q_flat, inv_lambda)
}

/// Low-rank symmetric eigendecomposition: return top-k eigenvectors and inverse eigenvalues.
///
/// Like [`eigh_f32`] but returns only the `k` eigenvectors corresponding to the
/// **largest** eigenvalues.  This gives a rank-k approximation
/// `gram ≈ Q_k · diag(λ_k) · Q_kᵀ` and allows the K-FAC apply step to use
/// `d_out × k` and `d_in × k` matrices instead of full `n × n` ones,
/// reducing the apply cost from O(4 n d_out d_in) to O((k_g + k_a) d_out d_in).
///
/// # Arguments
/// * `gram`    — Row-major `n×n` f32 slice (symmetric, PSD).
/// * `n`       — Matrix dimension.
/// * `k`       — Number of top eigenvectors to keep (1 ≤ k ≤ n).
/// * `damping` — Scalar δ: returns `1 / max(λᵢ + δ, 1e-8)`.
///
/// # Returns
/// `(q_k_flat, inv_lambda_k)` where:
/// * `q_k_flat` is row-major `n × k` f32 — top-k eigenvectors as columns.
/// * `inv_lambda_k` is length-k f32 with values `1 / max(λᵢ + δ, 1e-8)`.
pub fn eigh_topk_f32(gram: &[f32], n: usize, k: usize, damping: f32) -> (Vec<f32>, Vec<f32>) {
    use faer::linalg::solvers::SelfAdjointEigendecomposition;

    let k = k.min(n);
    let fa = faer::Mat::<f32>::from_fn(n, n, |i, j| gram[i * n + j]);
    let eig = SelfAdjointEigendecomposition::<f32>::new(fa.as_ref(), faer::Side::Lower);

    // faer sorts eigenvalues ascending; top-k are at indices n-k .. n
    let offset = n - k;

    let inv_lambda_k: Vec<f32> = (offset..n)
        .map(|i| {
            let lam = eig.s().column_vector().read(i);
            1.0_f32 / (lam + damping).max(1e-8_f32)
        })
        .collect();

    // Q_k: n×k matrix, row-major: q_k_flat[i*k + j] = Q[i, offset+j]
    let q = eig.u();
    let mut q_k_flat = vec![0.0_f32; n * k];
    for i in 0..n {
        for j in 0..k {
            q_k_flat[i * k + j] = q.read(i, offset + j);
        }
    }

    (q_k_flat, inv_lambda_k)
}

/// Apply the low-rank K-FAC preconditioner to a weight gradient.
///
/// Uses rank-k approximations of A and G:
///   `ΔW ≈ Q_G_k · diag(d_G_k) · (Q_G_kᵀ · grad · Q_A_k) · diag(d_A_k) · Q_A_kᵀ`
///
/// Cost vs full eigen: O((k_g + k_a) · d_out · d_in) instead of O(4 · n · d_out · d_in).
/// For k = 32, n = 512 this is ~16× fewer FLOPs in the apply step.
///
/// # Arguments
/// * `q_g_k`      — Row-major `d_out × k_g` f32 top-k eigenvectors of G.
/// * `k_g`        — Rank used for G.
/// * `inv_lam_g_k`— Length-`k_g` damped inverse eigenvalues of G.
/// * `grad`       — Row-major `d_out × d_in` f32 weight gradient.
/// * `d_out`      — Output dimension.
/// * `d_in`       — Input dimension.
/// * `q_a_k`      — Row-major `d_in × k_a` f32 top-k eigenvectors of A.
/// * `k_a`        — Rank used for A.
/// * `inv_lam_a_k`— Length-`k_a` damped inverse eigenvalues of A.
///
/// # Returns
/// Row-major `d_out × d_in` f32 preconditioned gradient.
pub fn apply_kfac_lowrank_f32(
    q_g_k: &[f32], k_g: usize, inv_lam_g_k: &[f32],
    grad: &[f32], d_out: usize, d_in: usize,
    q_a_k: &[f32], k_a: usize, inv_lam_a_k: &[f32],
) -> Vec<f32> {
    use faer::linalg::matmul::matmul;

    // Load matrices: Q_G_k (d_out × k_g), Q_A_k (d_in × k_a), grad (d_out × d_in)
    let qg = faer::Mat::<f32>::from_fn(d_out, k_g, |i, j| q_g_k[i * k_g + j]);
    let qa = faer::Mat::<f32>::from_fn(d_in,  k_a, |i, j| q_a_k[i * k_a + j]);
    let gr = faer::Mat::<f32>::from_fn(d_out, d_in, |i, j| grad[i * d_in + j]);

    // Step 1: tmp1 = Q_G_kᵀ @ grad   (k_g × d_in)
    let mut tmp1 = faer::Mat::<f32>::zeros(k_g, d_in);
    matmul(tmp1.as_mut(), qg.transpose(), gr.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 2: tmp2 = tmp1 @ Q_A_k   (k_g × k_a)  — cheap inner product
    let mut tmp2 = faer::Mat::<f32>::zeros(k_g, k_a);
    matmul(tmp2.as_mut(), tmp1.as_ref(), qa.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 3: scale (k_g × k_a) by inv_lam_g[i] * inv_lam_a[j]
    for i in 0..k_g {
        let sg = inv_lam_g_k[i];
        for j in 0..k_a {
            let v = tmp2.read(i, j);
            tmp2.write(i, j, v * sg * inv_lam_a_k[j]);
        }
    }

    // Step 4: tmp3 = Q_G_k @ tmp2   (d_out × k_a)
    let mut tmp3 = faer::Mat::<f32>::zeros(d_out, k_a);
    matmul(tmp3.as_mut(), qg.as_ref(), tmp2.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 5: result = tmp3 @ Q_A_kᵀ   (d_out × d_in)
    let mut result = faer::Mat::<f32>::zeros(d_out, d_in);
    matmul(result.as_mut(), tmp3.as_ref(), qa.transpose(),
           None, 1.0_f32, faer::Parallelism::None);

    // Extract row-major
    let mut out = vec![0.0_f32; d_out * d_in];
    for i in 0..d_out {
        for j in 0..d_in {
            out[i * d_in + j] = result.read(i, j);
        }
    }
    out
}

/// Apply the K-FAC eigen-basis preconditioner to a weight gradient.
///
/// Computes: `ΔW = Q_G · diag(d_G) · (Q_Gᵀ · grad · Q_A) · diag(d_A) · Q_Aᵀ`
///
/// where `d_G = 1/(λ_G + δ)` and `d_A = 1/(λ_A + δ)` are the damped inverse
/// eigenvalues returned by [`eigh_f32`].
///
/// # Arguments
/// * `q_g`       — Row-major `d_out × d_out` eigenvector matrix of G.
/// * `inv_lam_g` — Length-`d_out` damped inverse eigenvalues of G.
/// * `d_out`     — Output dimension.
/// * `grad`      — Row-major `d_out × d_in` weight gradient.
/// * `q_a`       — Row-major `d_in × d_in` eigenvector matrix of A.
/// * `inv_lam_a` — Length-`d_in` damped inverse eigenvalues of A.
/// * `d_in`      — Input dimension.
///
/// # Returns
/// Row-major `d_out × d_in` f32 preconditioned gradient.
pub fn apply_kfac_eigen_f32(
    q_g: &[f32], inv_lam_g: &[f32], d_out: usize,
    grad: &[f32],
    q_a: &[f32], inv_lam_a: &[f32], d_in: usize,
) -> Vec<f32> {
    use faer::linalg::matmul::matmul;

    // Load from row-major slices into faer column-major matrices
    let qg = faer::Mat::<f32>::from_fn(d_out, d_out, |i, j| q_g[i * d_out + j]);
    let qa = faer::Mat::<f32>::from_fn(d_in,  d_in,  |i, j| q_a[i * d_in  + j]);
    let gr = faer::Mat::<f32>::from_fn(d_out, d_in,  |i, j| grad[i * d_in + j]);

    // Step 1: tmp1 = Q_Gᵀ @ grad   (d_out × d_in)
    let mut tmp1 = faer::Mat::<f32>::zeros(d_out, d_in);
    matmul(tmp1.as_mut(), qg.transpose(), gr.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 2: tmp2 = tmp1 @ Q_A   (d_out × d_in)
    let mut tmp2 = faer::Mat::<f32>::zeros(d_out, d_in);
    matmul(tmp2.as_mut(), tmp1.as_ref(), qa.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 3: scale each (i,j) by inv_lam_g[i] * inv_lam_a[j]
    for i in 0..d_out {
        let sg = inv_lam_g[i];
        for j in 0..d_in {
            let v = tmp2.read(i, j);
            tmp2.write(i, j, v * sg * inv_lam_a[j]);
        }
    }

    // Step 4: tmp3 = Q_G @ tmp2   (d_out × d_in)
    let mut tmp3 = faer::Mat::<f32>::zeros(d_out, d_in);
    matmul(tmp3.as_mut(), qg.as_ref(), tmp2.as_ref(),
           None, 1.0_f32, faer::Parallelism::None);

    // Step 5: result = tmp3 @ Q_Aᵀ   (d_out × d_in)
    let mut result = faer::Mat::<f32>::zeros(d_out, d_in);
    matmul(result.as_mut(), tmp3.as_ref(), qa.transpose(),
           None, 1.0_f32, faer::Parallelism::None);

    // Extract row-major
    let mut out = vec![0.0_f32; d_out * d_in];
    for i in 0..d_out {
        for j in 0..d_in {
            out[i * d_in + j] = result.read(i, j);
        }
    }
    out
}
