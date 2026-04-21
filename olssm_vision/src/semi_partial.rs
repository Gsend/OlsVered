//! Semi-partial correlation extraction and gauge/degeneracy diagnostics.
//!
//! Background (Madar 2015, Madar & Batista 2023):
//! for a symmetric positive-definite matrix `A`, form the Cholesky factor
//! `L` such that `A = L Lᵀ`. After diagonal whitening `D^{-1/2} A D^{-1/2}`
//! the Cholesky factor has entries that are **semi-partial correlations**:
//! `L̃[i, j]` is the correlation of variable `i` with variable `j` after
//! controlling for variables `1..j-1`.
//!
//! Two operational consequences that matter for BA / SLAM:
//!
//! 1. Each diagonal pivot `L̃[j, j]` is the standard deviation of variable
//!    `j` *after* conditioning on the earlier variables. A pivot near zero
//!    says variable `j` carries no unique information once the earlier
//!    variables are known — i.e., a **gauge direction** or an **unobservable
//!    parameter**.
//!
//! 2. Semi-partial correlations give a principled drop tolerance for
//!    incomplete-Cholesky preconditioners: dropping `L̃[i, j]` entries
//!    below a correlation threshold preserves the statistically strongest
//!    conditional dependencies.
//!
//! This module is analytic / diagnostic. Phase 2 will swap the inner
//! Cholesky for Madar's closed-form entry-wise evaluation to unlock
//! column-streaming memory savings.

use crate::error::VisionError;
use nalgebra::{Cholesky, DMatrix, DVector};

/// Diagonal-whitened Cholesky factor of `a` plus its diagonal scales.
///
/// Returns `(L̃, d)` where `d[i] = sqrt(a[i,i])` and `L̃` is the lower-triangular
/// Cholesky of `D^{-1/2} a D^{-1/2}`. The full original Cholesky `L` of `a`
/// can be recovered as `L = D L̃` (diagonal scaling from the left).
pub fn whitened_cholesky(a: &DMatrix<f64>) -> Result<(DMatrix<f64>, DVector<f64>), VisionError> {
    let n = a.nrows();
    if a.ncols() != n {
        return Err(VisionError::DimensionMismatch {
            expected: n,
            got: a.ncols(),
            what: "A must be square",
        });
    }
    let mut d = DVector::<f64>::zeros(n);
    for i in 0..n {
        let di = a[(i, i)];
        if di <= 0.0 {
            return Err(VisionError::RankDeficientS);
        }
        d[i] = di.sqrt();
    }

    let mut r = a.clone();
    for i in 0..n {
        for j in 0..n {
            r[(i, j)] /= d[i] * d[j];
        }
    }

    let chol = Cholesky::new(r).ok_or(VisionError::RankDeficientS)?;
    Ok((chol.l(), d))
}

/// Entry-wise semi-partial correlation at `(i, j)` with `i >= j`.
///
/// This is the `(i, j)` entry of the whitened Cholesky factor, interpreted
/// as the correlation of variable `i` with variable `j` after conditioning
/// on variables `0..j`.
pub fn semi_partial(l_whitened: &DMatrix<f64>, i: usize, j: usize) -> f64 {
    if j > i {
        return 0.0;
    }
    l_whitened[(i, j)]
}

/// Gauge / degeneracy report for a BA information matrix.
///
/// Each diagonal pivot below `threshold` identifies a parameter direction
/// that is near-unobservable given the current measurements. Typical causes:
///
/// * global SE(3) gauge freedom of the whole reconstruction (7 directions),
/// * a single camera with insufficient parallax against the scene,
/// * a planar scene that collapses focal-length and Z-translation,
/// * pure-rotation motion that makes translation unobservable.
///
/// Returns a vector of `(column_index, pivot_magnitude)` for the offending
/// columns, in ascending order of pivot magnitude (weakest first).
pub fn detect_near_singular(
    l_whitened: &DMatrix<f64>,
    threshold: f64,
) -> Vec<(usize, f64)> {
    let n = l_whitened.nrows();
    let mut out: Vec<(usize, f64)> = (0..n)
        .map(|i| (i, l_whitened[(i, i)]))
        .filter(|(_, p)| *p < threshold)
        .collect();
    out.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    out
}

/// Vector of all diagonal pivots — useful as a single-shot observability
/// signature for the current LM iterate.
pub fn pivot_profile(l_whitened: &DMatrix<f64>) -> DVector<f64> {
    let n = l_whitened.nrows();
    let mut out = DVector::<f64>::zeros(n);
    for i in 0..n {
        out[i] = l_whitened[(i, i)];
    }
    out
}

/// Sparse fill-in pattern guided by semi-partial correlation magnitude.
///
/// Returns the (i, j) index pairs whose |L̃[i, j]| >= tolerance. Intended
/// as the sparsity template for an incomplete-Cholesky preconditioner: the
/// strongest conditional dependencies are retained, weaker ones dropped.
pub fn sparsity_template(l_whitened: &DMatrix<f64>, tolerance: f64) -> Vec<(usize, usize)> {
    let n = l_whitened.nrows();
    let mut out = Vec::new();
    for i in 0..n {
        for j in 0..=i {
            if l_whitened[(i, j)].abs() >= tolerance {
                out.push((i, j));
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn whitened_cholesky_reproduces_a() {
        let m = DMatrix::<f64>::from_row_slice(3, 3, &[
            4.0, 1.0, 0.5,
            1.0, 9.0, 0.2,
            0.5, 0.2, 16.0,
        ]);
        let (lw, d) = whitened_cholesky(&m).unwrap();
        // Reconstruct: L = diag(d) * Lw; A ?= L Lᵀ
        let mut lfull = lw.clone();
        for i in 0..3 {
            for j in 0..3 {
                lfull[(i, j)] *= d[i];
            }
        }
        let reconstructed = &lfull * lfull.transpose();
        for i in 0..3 {
            for j in 0..3 {
                assert_relative_eq!(reconstructed[(i, j)], m[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn detect_near_singular_flags_gauge_column() {
        // Construct a 4×4 where the last column/row has almost no independent
        // variance: variable 3 is nearly a copy of variable 0.
        let mut a = DMatrix::<f64>::identity(4, 4) * 5.0;
        a[(0, 3)] = 5.0 - 1e-5;
        a[(3, 0)] = 5.0 - 1e-5;
        a[(3, 3)] = 5.0;
        let (lw, _) = whitened_cholesky(&a).unwrap();
        let weak = detect_near_singular(&lw, 0.01);
        assert!(weak.iter().any(|(idx, _)| *idx == 3));
    }

    #[test]
    fn sparsity_template_respects_tolerance() {
        let m = DMatrix::<f64>::from_row_slice(3, 3, &[
            4.0, 0.01, 0.001,
            0.01, 4.0, 0.01,
            0.001, 0.01, 4.0,
        ]);
        let (lw, _) = whitened_cholesky(&m).unwrap();
        let pattern = sparsity_template(&lw, 0.05);
        // Diagonals are 1.0 after whitening, always retained
        assert!(pattern.contains(&(0, 0)));
        assert!(pattern.contains(&(1, 1)));
        assert!(pattern.contains(&(2, 2)));
        // Very-weak off-diagonals should be dropped
        assert!(!pattern.contains(&(2, 0)));
    }
}
