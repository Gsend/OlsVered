//! Schur complement for the bundle-adjustment reduced-camera system.
//!
//! Given the BA Hessian block-structured as
//! ```text
//!     H = [ U   B ]      g = [ g_c ]
//!         [ Bᵀ  V ]          [ g_p ]
//! ```
//! with `U ∈ R^{6·Nc × 6·Nc}`, `V` block-diagonal of 3×3 point blocks,
//! and `B ∈ R^{6·Nc × 3·Np}`, this module forms the reduced camera system
//! ```text
//!     S  δ_c = g̃
//!     S  = U − B V⁻¹ Bᵀ
//!     g̃  = g_c − B V⁻¹ g_p
//! ```
//! `V⁻¹` is cheap because `V` is block-diagonal with 3×3 blocks (trivial
//! closed-form inverse).
//!
//! This first implementation is dense end-to-end. A sparse path (Phase 2)
//! would exploit the sparsity of `B` over the co-visibility graph.

use crate::error::VisionError;
use nalgebra::{DMatrix, DVector, Matrix3};

/// In-place 3×3 inverse via cofactor expansion. Returns `None` if singular.
fn invert_3x3(m: &DMatrix<f64>, i0: usize, j0: usize) -> Option<Matrix3<f64>> {
    let a = Matrix3::new(
        m[(i0, j0)], m[(i0, j0 + 1)], m[(i0, j0 + 2)],
        m[(i0 + 1, j0)], m[(i0 + 1, j0 + 1)], m[(i0 + 1, j0 + 2)],
        m[(i0 + 2, j0)], m[(i0 + 2, j0 + 1)], m[(i0 + 2, j0 + 2)],
    );
    a.try_inverse()
}

/// Result of Schur elimination: reduced camera system plus the cached
/// `V⁻¹` and the off-diagonal `B` needed to back-substitute for point
/// updates once `δ_c` is solved.
#[derive(Debug, Clone)]
pub struct SchurSystem {
    pub s: DMatrix<f64>,              // reduced camera Hessian (6Nc × 6Nc)
    pub g_tilde: DVector<f64>,        // reduced RHS (6Nc)
    pub v_inv_blocks: Vec<Matrix3<f64>>, // per-point V_i⁻¹ blocks
    pub b: DMatrix<f64>,              // original B matrix (6Nc × 3Np), cached
    pub g_p: DVector<f64>,            // original point RHS (3Np)
}

impl SchurSystem {
    /// Back-substitute point updates: given the camera update `δ_c`, compute
    /// `δ_p = V⁻¹ (g_p − Bᵀ δ_c)`.
    pub fn backsubstitute_points(&self, delta_c: &DVector<f64>) -> DVector<f64> {
        let n_points = self.v_inv_blocks.len();
        let rhs_p = &self.g_p - self.b.transpose() * delta_c; // 3Np
        let mut delta_p = DVector::<f64>::zeros(3 * n_points);
        for i in 0..n_points {
            let base = 3 * i;
            let rhs = nalgebra::Vector3::new(rhs_p[base], rhs_p[base + 1], rhs_p[base + 2]);
            let dp = self.v_inv_blocks[i] * rhs;
            delta_p[base] = dp.x;
            delta_p[base + 1] = dp.y;
            delta_p[base + 2] = dp.z;
        }
        delta_p
    }
}

/// Build the Schur-reduced camera system.
///
/// Arguments
/// * `u` — camera block, `(6 Nc) × (6 Nc)`
/// * `v` — point block stored as a dense block-diagonal, `(3 Np) × (3 Np)`
/// * `b` — coupling, `(6 Nc) × (3 Np)`
/// * `g_c` — camera RHS, length `6 Nc`
/// * `g_p` — point RHS, length `3 Np`
/// * `damping` — Levenberg-Marquardt damping added to the diagonal of both
///   `U` and each `V_i` before elimination.
pub fn build_schur(
    u: &DMatrix<f64>,
    v: &DMatrix<f64>,
    b: &DMatrix<f64>,
    g_c: &DVector<f64>,
    g_p: &DVector<f64>,
    damping: f64,
) -> Result<SchurSystem, VisionError> {
    let n_cam_dof = u.nrows();
    let n_pt_dof = v.nrows();

    if u.ncols() != n_cam_dof {
        return Err(VisionError::DimensionMismatch {
            expected: n_cam_dof,
            got: u.ncols(),
            what: "U must be square",
        });
    }
    if v.ncols() != n_pt_dof {
        return Err(VisionError::DimensionMismatch {
            expected: n_pt_dof,
            got: v.ncols(),
            what: "V must be square",
        });
    }
    if n_pt_dof % 3 != 0 {
        return Err(VisionError::DimensionMismatch {
            expected: 3,
            got: n_pt_dof % 3,
            what: "V rows must be multiple of 3",
        });
    }
    if b.nrows() != n_cam_dof || b.ncols() != n_pt_dof {
        return Err(VisionError::DimensionMismatch {
            expected: n_cam_dof * n_pt_dof,
            got: b.nrows() * b.ncols(),
            what: "B must be (6Nc) × (3Np)",
        });
    }
    if g_c.len() != n_cam_dof {
        return Err(VisionError::DimensionMismatch {
            expected: n_cam_dof,
            got: g_c.len(),
            what: "g_c length",
        });
    }
    if g_p.len() != n_pt_dof {
        return Err(VisionError::DimensionMismatch {
            expected: n_pt_dof,
            got: g_p.len(),
            what: "g_p length",
        });
    }

    let n_points = n_pt_dof / 3;

    // Damped V with λI added per 3×3 block diagonal, then inverted.
    let mut v_inv_blocks: Vec<Matrix3<f64>> = Vec::with_capacity(n_points);
    for i in 0..n_points {
        let base = 3 * i;
        let mut block = Matrix3::new(
            v[(base, base)], v[(base, base + 1)], v[(base, base + 2)],
            v[(base + 1, base)], v[(base + 1, base + 1)], v[(base + 1, base + 2)],
            v[(base + 2, base)], v[(base + 2, base + 1)], v[(base + 2, base + 2)],
        );
        block[(0, 0)] += damping;
        block[(1, 1)] += damping;
        block[(2, 2)] += damping;
        let inv = block
            .try_inverse()
            .ok_or(VisionError::SingularPointBlock { index: i })?;
        v_inv_blocks.push(inv);
    }

    // Compute B V⁻¹: apply each 3-column slab through the matching V_i⁻¹.
    let mut b_vinv = DMatrix::<f64>::zeros(n_cam_dof, n_pt_dof);
    for i in 0..n_points {
        let base = 3 * i;
        // B columns [base..base+3] get multiplied on the right by V_i⁻¹.
        let b_slab = b.columns(base, 3);
        let vinv = &v_inv_blocks[i];
        // Convert Matrix3 to DMatrix for multiplication
        let mut vinv_dyn = DMatrix::<f64>::zeros(3, 3);
        for r in 0..3 {
            for c in 0..3 {
                vinv_dyn[(r, c)] = vinv[(r, c)];
            }
        }
        let product = b_slab * &vinv_dyn;
        for (idx, col) in (base..base + 3).enumerate() {
            b_vinv.column_mut(col).copy_from(&product.column(idx));
        }
    }

    // S = (U + λI) − (B V⁻¹) Bᵀ
    let mut s = u.clone();
    for i in 0..n_cam_dof {
        s[(i, i)] += damping;
    }
    s -= &b_vinv * b.transpose();

    // g̃ = g_c − (B V⁻¹) g_p
    let g_tilde = g_c - &b_vinv * g_p;

    Ok(SchurSystem {
        s,
        g_tilde,
        v_inv_blocks,
        b: b.clone(),
        g_p: g_p.clone(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn invert_3x3_of_identity_block() {
        let mut m = DMatrix::<f64>::identity(3, 3);
        m[(1, 2)] = 0.5;
        let inv = invert_3x3(&m, 0, 0).unwrap();
        // identity + upper-tri perturbation, check a couple entries
        assert_relative_eq!(inv[(0, 0)], 1.0, epsilon = 1e-12);
        assert_relative_eq!(inv[(1, 2)], -0.5, epsilon = 1e-12);
    }

    #[test]
    fn schur_matches_direct_inverse_on_toy_problem() {
        // 2 cameras × 6 DoF = 12 camera dims, 3 points × 3 = 9 point dims
        let nc = 12;
        let np_ = 9;

        // Construct a symmetric PD H by H = MᵀM + κI
        let m = DMatrix::<f64>::from_fn(nc + np_, nc + np_, |i, j| {
            (((i + 1) as f64).sin() * ((j + 2) as f64).cos()).abs() + 0.1
        });
        let mut h = m.transpose() * &m;
        // Zero off-diagonal entries inside V so V stays block-diagonal
        for i in nc..nc + np_ {
            for j in nc..nc + np_ {
                let ii = (i - nc) / 3;
                let jj = (j - nc) / 3;
                if ii != jj {
                    h[(i, j)] = 0.0;
                }
            }
        }
        for i in 0..nc + np_ {
            h[(i, i)] += 10.0;
        }

        let u = h.view((0, 0), (nc, nc)).into_owned();
        let v = h.view((nc, nc), (np_, np_)).into_owned();
        let b = h.view((0, nc), (nc, np_)).into_owned();

        let g_c = DVector::<f64>::from_fn(nc, |i, _| (i as f64).cos());
        let g_p = DVector::<f64>::from_fn(np_, |i, _| (i as f64 + 0.5).sin());

        let schur = build_schur(&u, &v, &b, &g_c, &g_p, 0.0).unwrap();

        // Reference: directly solve the full system and compare δ_c
        let g = {
            let mut g = DVector::<f64>::zeros(nc + np_);
            g.view_mut((0, 0), (nc, 1)).copy_from(&g_c);
            g.view_mut((nc, 0), (np_, 1)).copy_from(&g_p);
            g
        };
        let delta_full = h.clone().lu().solve(&g).expect("full solve failed");
        let delta_c_ref = delta_full.view((0, 0), (nc, 1)).into_owned();
        let delta_p_ref = delta_full.view((nc, 0), (np_, 1)).into_owned();

        let delta_c = schur.s.clone().lu().solve(&schur.g_tilde).expect("schur solve");
        let delta_p = schur.backsubstitute_points(&delta_c);

        for i in 0..nc {
            assert_relative_eq!(delta_c[i], delta_c_ref[i], epsilon = 1e-8);
        }
        for i in 0..np_ {
            assert_relative_eq!(delta_p[i], delta_p_ref[i], epsilon = 1e-8);
        }
    }
}
