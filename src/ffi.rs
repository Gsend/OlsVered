//! C ABI layer — `extern "C"` exports for C/C++ consumers.
//!
//! cbindgen reads this file to generate `include/olsvered.h`.
//!
//! **Memory layout:** all matrices are **column-major** (Fortran / C-style
//! `double*` arrays as used by BLAS/LAPACK).  Callers must allocate output
//! buffers to the sizes documented on each function.

use std::slice;

use nalgebra::{DMatrix, DVector};

use crate::algorithms;

// ---------------------------------------------------------------------------
// Status codes
// ---------------------------------------------------------------------------

/// Status codes returned by all `olsvered_*` FFI functions.
#[repr(C)]
pub enum OlsveredStatus {
    /// Success.
    Ok = 0,
    /// Row/column dimension mismatch between inputs.
    DimensionMismatch = 1,
    /// Zero pivot encountered — input is (near-)singular.
    ZeroPivot = 2,
    /// A required pointer argument was NULL.
    NullPointer = 3,
    /// Matrix is singular; solve returned no solution.
    SingularMatrix = 4,
}

impl From<algorithms::OlsveredError> for OlsveredStatus {
    fn from(e: algorithms::OlsveredError) -> Self {
        match e {
            algorithms::OlsveredError::DimensionMismatch { .. } => {
                OlsveredStatus::DimensionMismatch
            }
            algorithms::OlsveredError::ZeroPivot { .. } => OlsveredStatus::ZeroPivot,
            algorithms::OlsveredError::WeightDimension { .. } => {
                OlsveredStatus::DimensionMismatch
            }
            algorithms::OlsveredError::SingularMatrix => OlsveredStatus::SingularMatrix,
        }
    }
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/// Reconstruct a column-major DMatrix from a raw pointer + dimensions.
///
/// # Safety
/// Caller must ensure `ptr` points to at least `rows * cols` valid `f64`s.
unsafe fn mat_from_ptr(ptr: *const f64, rows: usize, cols: usize) -> DMatrix<f64> {
    let slice = slice::from_raw_parts(ptr, rows * cols);
    DMatrix::from_column_slice(rows, cols, slice)
}

/// Reconstruct a DVector from a raw pointer + length.
///
/// # Safety
/// Caller must ensure `ptr` points to at least `len` valid `f64`s.
unsafe fn vec_from_ptr(ptr: *const f64, len: usize) -> DVector<f64> {
    let slice = slice::from_raw_parts(ptr, len);
    DVector::from_column_slice(slice)
}

/// Copy a DMatrix (column-major) into a caller-allocated output buffer.
///
/// # Safety
/// `out` must point to at least `mat.nrows() * mat.ncols()` writable `f64`s.
unsafe fn mat_to_ptr(mat: &DMatrix<f64>, out: *mut f64) {
    let out_slice = slice::from_raw_parts_mut(out, mat.nrows() * mat.ncols());
    out_slice.copy_from_slice(mat.as_slice());
}

/// Copy a DVector into a caller-allocated output buffer.
///
/// # Safety
/// `out` must point to at least `vec.len()` writable `f64`s.
unsafe fn vec_to_ptr(vec: &DVector<f64>, out: *mut f64) {
    let out_slice = slice::from_raw_parts_mut(out, vec.len());
    out_slice.copy_from_slice(vec.as_slice());
}

// ---------------------------------------------------------------------------
// Algorithm 1 — Modified Cholesky
// ---------------------------------------------------------------------------

/// Algorithm 1: Modified Cholesky.
///
/// @param x        Column-major f64 array, shape (x_rows × x_cols)
/// @param x_rows   Number of rows in X (n samples)
/// @param x_cols   Number of columns in X (p predictors)
/// @param y        f64 array of length x_rows
/// @param c_out    Caller-allocated output buffer of size (x_cols+1)*(x_cols+1)
/// @return         OlsveredStatus::Ok on success
#[no_mangle]
pub unsafe extern "C" fn olsvered_modified_cholesky(
    x: *const f64,
    x_rows: usize,
    x_cols: usize,
    y: *const f64,
    c_out: *mut f64,
) -> OlsveredStatus {
    if x.is_null() || y.is_null() || c_out.is_null() {
        return OlsveredStatus::NullPointer;
    }
    let xm = mat_from_ptr(x, x_rows, x_cols);
    let yv = vec_from_ptr(y, x_rows);
    match algorithms::modified_cholesky(&xm, &yv) {
        Ok(c) => {
            mat_to_ptr(&c, c_out);
            OlsveredStatus::Ok
        }
        Err(e) => OlsveredStatus::from(e),
    }
}

// ---------------------------------------------------------------------------
// Back-substitution
// ---------------------------------------------------------------------------

/// Back-substitute C to recover OLS beta coefficients.
///
/// @param c        Column-major f64 array, shape (dim × dim)
/// @param dim      Dimension of C (= p+1)
/// @param beta_out Caller-allocated output buffer of length (dim-1)
/// @return         OlsveredStatus::Ok on success
#[no_mangle]
pub unsafe extern "C" fn olsvered_back_substitute(
    c: *const f64,
    dim: usize,
    beta_out: *mut f64,
) -> OlsveredStatus {
    if c.is_null() || beta_out.is_null() {
        return OlsveredStatus::NullPointer;
    }
    let cm = mat_from_ptr(c, dim, dim);
    match algorithms::back_substitute(&cm) {
        Ok(beta) => {
            vec_to_ptr(&beta, beta_out);
            OlsveredStatus::Ok
        }
        Err(e) => OlsveredStatus::from(e),
    }
}

// ---------------------------------------------------------------------------
// Combined OLS solver
// ---------------------------------------------------------------------------

/// Full OLS solver (Algorithm 1 + back-substitution).
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param y        f64 array of length x_rows
/// @param beta_out Caller-allocated output buffer of length x_cols
/// @return         OlsveredStatus::Ok on success
#[no_mangle]
pub unsafe extern "C" fn olsvered_solve_ols(
    x: *const f64,
    x_rows: usize,
    x_cols: usize,
    y: *const f64,
    beta_out: *mut f64,
) -> OlsveredStatus {
    if x.is_null() || y.is_null() || beta_out.is_null() {
        return OlsveredStatus::NullPointer;
    }
    let xm = mat_from_ptr(x, x_rows, x_cols);
    let yv = vec_from_ptr(y, x_rows);
    match algorithms::solve_ols(&xm, &yv) {
        Ok(beta) => {
            vec_to_ptr(&beta, beta_out);
            OlsveredStatus::Ok
        }
        Err(e) => OlsveredStatus::from(e),
    }
}

// ---------------------------------------------------------------------------
// Algorithm 2 — Simplified Gram-Schmidt
// ---------------------------------------------------------------------------

/// Algorithm 2: Simplified (non-normalised) Gram-Schmidt orthogonalisation.
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param q_out    Caller-allocated output buffer (x_rows * x_cols)
/// @return         OlsveredStatus::Ok on success
#[no_mangle]
pub unsafe extern "C" fn olsvered_simplified_gram_schmidt(
    x: *const f64,
    x_rows: usize,
    x_cols: usize,
    q_out: *mut f64,
) -> OlsveredStatus {
    if x.is_null() || q_out.is_null() {
        return OlsveredStatus::NullPointer;
    }
    let xm = mat_from_ptr(x, x_rows, x_cols);
    match algorithms::simplified_gram_schmidt(&xm) {
        Ok(q) => {
            mat_to_ptr(&q, q_out);
            OlsveredStatus::Ok
        }
        Err(e) => OlsveredStatus::from(e),
    }
}

// ---------------------------------------------------------------------------
// Algorithm 3 — Weighted Generalised Inverse
// ---------------------------------------------------------------------------

/// Algorithm 3: Weighted generalised inverse (XᵀWX)⁻¹ Xᵀ W.
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param w        Column-major f64 array (x_rows × x_rows) — weight matrix
/// @param g_out    Caller-allocated output buffer (x_cols * x_rows)
/// @return         OlsveredStatus::Ok on success
#[no_mangle]
pub unsafe extern "C" fn olsvered_weighted_generalized_inverse(
    x: *const f64,
    x_rows: usize,
    x_cols: usize,
    w: *const f64,
    g_out: *mut f64,
) -> OlsveredStatus {
    if x.is_null() || w.is_null() || g_out.is_null() {
        return OlsveredStatus::NullPointer;
    }
    let xm = mat_from_ptr(x, x_rows, x_cols);
    let wm = mat_from_ptr(w, x_rows, x_rows);
    match algorithms::weighted_generalized_inverse(&xm, &wm) {
        Ok(g) => {
            mat_to_ptr(&g, g_out);
            OlsveredStatus::Ok
        }
        Err(e) => OlsveredStatus::from(e),
    }
}
