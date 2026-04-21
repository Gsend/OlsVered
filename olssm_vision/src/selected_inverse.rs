//! Selected-inverse computation for SLAM / BA covariance recovery.
//!
//! Given `A = L Lᵀ` (Cholesky of the reduced camera Hessian `S`), this
//! module computes selected entries or columns of `A⁻¹` without forming
//! the full dense inverse. Two entry points:
//!
//! * [`column`] — solve `A x = e_j` via two triangular solves, returning
//!   the full `j`-th column of `A⁻¹` in `O(n²)` flops with `O(n)` memory.
//!
//! * [`diagonal`] — Takahashi backward recursion on the Cholesky factor,
//!   returning `diag(A⁻¹)` in `O(n²)` flops (no dense inverse materialised).
//!
//! * [`block`] — extract a `k × k` symmetric block of `A⁻¹` at the given
//!   indices. `O(k · n²)` via repeated column solves.
//!
//! These are the primitives called by the `slam.query_covariance(...)`
//! Python API to serve per-camera / per-landmark uncertainty queries
//! on demand, at a fraction of the cost of `A⁻¹` in full.

use crate::error::VisionError;
use nalgebra::{DMatrix, DVector};

/// Solve `L y = b` in-place via forward substitution. `L` must be lower
/// triangular with non-zero diagonal.
fn forward_sub(l: &DMatrix<f64>, b: &DVector<f64>) -> Result<DVector<f64>, VisionError> {
    let n = l.nrows();
    if l.ncols() != n || b.len() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: b.len(),
            what: "forward_sub dims",
        });
    }
    let mut y = DVector::<f64>::zeros(n);
    for i in 0..n {
        let mut sum = b[i];
        for k in 0..i {
            sum -= l[(i, k)] * y[k];
        }
        let d = l[(i, i)];
        if d.abs() < 1e-300 {
            return Err(VisionError::RankDeficientS);
        }
        y[i] = sum / d;
    }
    Ok(y)
}

/// Solve `Lᵀ x = y` in-place via back substitution.
fn back_sub_transpose(l: &DMatrix<f64>, y: &DVector<f64>) -> Result<DVector<f64>, VisionError> {
    let n = l.nrows();
    if l.ncols() != n || y.len() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: y.len(),
            what: "back_sub dims",
        });
    }
    let mut x = DVector::<f64>::zeros(n);
    for i in (0..n).rev() {
        let mut sum = y[i];
        for k in (i + 1)..n {
            sum -= l[(k, i)] * x[k];
        }
        let d = l[(i, i)];
        if d.abs() < 1e-300 {
            return Err(VisionError::RankDeficientS);
        }
        x[i] = sum / d;
    }
    Ok(x)
}

/// `j`-th column of `A⁻¹` where `A = L Lᵀ`.
pub fn column(l: &DMatrix<f64>, j: usize) -> Result<DVector<f64>, VisionError> {
    let n = l.nrows();
    if j >= n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: j,
            what: "column index",
        });
    }
    let mut e = DVector::<f64>::zeros(n);
    e[j] = 1.0;
    let y = forward_sub(l, &e)?;
    back_sub_transpose(l, &y)
}

/// Diagonal of `A⁻¹` via Takahashi backward recursion.
///
/// Reference: Takahashi, Fagan & Chen, "Formation of a sparse bus impedance
/// matrix" (1973); also Erisman & Tinney's algorithm. We implement the dense
/// full-matrix variant: each output entry costs `O(n − i)` work, total `O(n²)`.
///
/// For `A = L Lᵀ` with lower-triangular `L`:
/// ```text
///   Σ_{n-1, n-1} = 1 / L_{n-1, n-1}²
///   for i = n-2 downto 0:
///       Σ_{i, i} = 1/L_ii²
///                − (1/L_ii) · Σ_{k=i+1..n-1} L_{k, i} · Σ_{k, i}
///       (Σ_{k, i}  for k > i  computed en route)
/// ```
pub fn diagonal(l: &DMatrix<f64>) -> Result<DVector<f64>, VisionError> {
    let n = l.nrows();
    if l.ncols() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: l.ncols(),
            what: "L must be square",
        });
    }
    // Dense Σ storage for the recursion, lower-triangle only.
    let mut sigma = DMatrix::<f64>::zeros(n, n);

    // Starting point: bottom-right 1×1 block.
    let d_last = l[(n - 1, n - 1)];
    if d_last.abs() < 1e-300 {
        return Err(VisionError::RankDeficientS);
    }
    sigma[(n - 1, n - 1)] = 1.0 / (d_last * d_last);

    // Walk upward.
    for i in (0..n - 1).rev() {
        let lii = l[(i, i)];
        if lii.abs() < 1e-300 {
            return Err(VisionError::RankDeficientS);
        }

        // Off-diagonal Σ[k, i] for k > i.
        // Σ[k, i] = − (1/L_ii) · Σ_{j=i+1..n-1} L_{j, i} · Σ[k, j]  (for k ≥ j)
        //                                    + L_{j, i} · Σ[j, k]  (for k < j)
        // combined as:
        //        = − (1/L_ii) · Σ_{j=i+1..n-1} L_{j, i} · Σ_sym[k, j]
        // where Σ_sym[k, j] reads the lower-triangle entry (max(k,j), min(k,j)).
        for k in (i + 1)..n {
            let mut acc = 0.0;
            for j in (i + 1)..n {
                let ljl = l[(j, i)];
                if ljl == 0.0 {
                    continue;
                }
                let (r, c) = if k >= j { (k, j) } else { (j, k) };
                acc += ljl * sigma[(r, c)];
            }
            sigma[(k, i)] = -acc / lii;
        }

        // Diagonal Σ[i, i].
        let mut acc = 0.0;
        for k in (i + 1)..n {
            acc += l[(k, i)] * sigma[(k, i)];
        }
        sigma[(i, i)] = 1.0 / (lii * lii) - acc / lii;
    }

    let mut diag = DVector::<f64>::zeros(n);
    for i in 0..n {
        diag[i] = sigma[(i, i)];
    }
    Ok(diag)
}

/// `k × k` symmetric block of `A⁻¹` at `indices`. Complexity `O(k · n²)`.
pub fn block(l: &DMatrix<f64>, indices: &[usize]) -> Result<DMatrix<f64>, VisionError> {
    let k = indices.len();
    let n = l.nrows();
    let mut out = DMatrix::<f64>::zeros(k, k);
    for (a, &ja) in indices.iter().enumerate() {
        if ja >= n {
            return Err(VisionError::DimensionMismatch {
                expected: n,
                got: ja,
                what: "block index",
            });
        }
        let col = column(l, ja)?;
        for (b, &ib) in indices.iter().enumerate() {
            out[(b, a)] = col[ib];
        }
    }
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    fn random_spd(n: usize, seed: f64) -> DMatrix<f64> {
        let m = DMatrix::<f64>::from_fn(n, n, |i, j| {
            (((i + 1) as f64 + seed).sin() * ((j + 2) as f64 + seed).cos()).abs() + 0.1
        });
        let mut a = m.transpose() * &m;
        for i in 0..n {
            a[(i, i)] += n as f64;
        }
        a
    }

    #[test]
    fn column_matches_naive_inverse() {
        let n = 7;
        let a = random_spd(n, 0.3);
        let l = nalgebra::Cholesky::new(a.clone()).unwrap().l();
        let a_inv = a.try_inverse().unwrap();
        for j in 0..n {
            let col = column(&l, j).unwrap();
            for i in 0..n {
                assert_relative_eq!(col[i], a_inv[(i, j)], epsilon = 1e-9);
            }
        }
    }

    #[test]
    fn diagonal_matches_naive_inverse() {
        let n = 8;
        let a = random_spd(n, 1.7);
        let l = nalgebra::Cholesky::new(a.clone()).unwrap().l();
        let a_inv = a.try_inverse().unwrap();
        let diag = diagonal(&l).unwrap();
        for i in 0..n {
            assert_relative_eq!(diag[i], a_inv[(i, i)], epsilon = 1e-9);
        }
    }

    #[test]
    fn block_extracts_3x3_point_covariance() {
        let n = 10;
        let a = random_spd(n, 2.5);
        let l = nalgebra::Cholesky::new(a.clone()).unwrap().l();
        let a_inv = a.try_inverse().unwrap();
        let idx = vec![3, 4, 5];
        let blk = block(&l, &idx).unwrap();
        for (a_i, &ii) in idx.iter().enumerate() {
            for (b_i, &jj) in idx.iter().enumerate() {
                assert_relative_eq!(blk[(a_i, b_i)], a_inv[(ii, jj)], epsilon = 1e-9);
            }
        }
    }
}
