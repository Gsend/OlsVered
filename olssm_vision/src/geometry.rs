//! Geometry primitives for BA / SLAM — SE(3) pose, pinhole intrinsics,
//! point projection and analytic Jacobians.
//!
//! Conventions:
//! * SE(3) tangent order is `[translation (3); rotation (3)]` (Forster/GTSAM order).
//! * All storage is dynamic (`DMatrix<f64>` / `DVector<f64>`) for API consistency
//!   with the `olssm` core crate. Performance-critical sizes can be specialized later.
//! * Right-multiplicative perturbation model for rotations.
//!
//! This module is intentionally minimal — enough to build and test the Schur
//! complement and selected-inverse machinery against synthetic BA scenes.
//! It is **not** a replacement for a full SO(3)/SE(3) Lie-group library.

use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3};

/// SE(3) pose: rotation matrix plus translation vector.
#[derive(Debug, Clone)]
pub struct SE3 {
    pub rotation: Matrix3<f64>,
    pub translation: Vector3<f64>,
}

impl SE3 {
    pub fn identity() -> Self {
        Self {
            rotation: Matrix3::identity(),
            translation: Vector3::zeros(),
        }
    }

    /// Apply the transform to a world-frame point: `p_cam = R p + t`.
    pub fn transform(&self, world_pt: &Vector3<f64>) -> Vector3<f64> {
        self.rotation * world_pt + self.translation
    }

    /// Inverse pose: `R^T`, `-R^T t`.
    pub fn inverse(&self) -> Self {
        let rt = self.rotation.transpose();
        Self {
            rotation: rt,
            translation: -(rt * self.translation),
        }
    }
}

/// Pinhole intrinsics: focal lengths and principal point in pixels.
#[derive(Debug, Clone, Copy)]
pub struct PinholeIntrinsics {
    pub fx: f64,
    pub fy: f64,
    pub cx: f64,
    pub cy: f64,
}

impl PinholeIntrinsics {
    pub fn new(fx: f64, fy: f64, cx: f64, cy: f64) -> Self {
        Self { fx, fy, cx, cy }
    }
}

/// Skew-symmetric (`[v]_×`) matrix for vector `v`.
#[inline]
fn skew(v: &Vector3<f64>) -> Matrix3<f64> {
    Matrix3::new(
        0.0, -v.z, v.y,
        v.z, 0.0, -v.x,
        -v.y, v.x, 0.0,
    )
}

/// Project a world-frame point through `pose` and `K`.
///
/// Returns `None` if the point falls behind the camera (`z <= 0`).
pub fn project(
    pose: &SE3,
    k: &PinholeIntrinsics,
    world_pt: &Vector3<f64>,
) -> Option<Vector2<f64>> {
    let p_c = pose.transform(world_pt);
    if p_c.z <= 0.0 {
        return None;
    }
    Some(Vector2::new(
        k.fx * p_c.x / p_c.z + k.cx,
        k.fy * p_c.y / p_c.z + k.cy,
    ))
}

/// Analytic Jacobian of the projection w.r.t. the 6-DoF pose tangent and
/// the 3-vector world point.
///
/// Returns `(J_pose, J_point)` where
/// * `J_pose` is 2×6 with columns `[d/dt (3), d/dω (3)]`,
/// * `J_point` is 2×3 in world-frame coordinates.
///
/// Returns `None` if the point is behind the camera.
pub fn projection_jacobian(
    pose: &SE3,
    k: &PinholeIntrinsics,
    world_pt: &Vector3<f64>,
) -> Option<(DMatrix<f64>, DMatrix<f64>)> {
    let p_c = pose.transform(world_pt);
    if p_c.z <= 0.0 {
        return None;
    }
    let z_inv = 1.0 / p_c.z;
    let z_inv2 = z_inv * z_inv;

    // d(u,v) / d(p_c)  — 2×3
    let mut dpix_dpc = DMatrix::<f64>::zeros(2, 3);
    dpix_dpc[(0, 0)] = k.fx * z_inv;
    dpix_dpc[(0, 1)] = 0.0;
    dpix_dpc[(0, 2)] = -k.fx * p_c.x * z_inv2;
    dpix_dpc[(1, 0)] = 0.0;
    dpix_dpc[(1, 1)] = k.fy * z_inv;
    dpix_dpc[(1, 2)] = -k.fy * p_c.y * z_inv2;

    // d(p_c) / d(pose)  — 3×6 block [I_3 | -[p_c]_×]
    let mut dpc_dpose = DMatrix::<f64>::zeros(3, 6);
    for i in 0..3 {
        dpc_dpose[(i, i)] = 1.0;
    }
    let sk = skew(&p_c);
    for i in 0..3 {
        for j in 0..3 {
            dpc_dpose[(i, 3 + j)] = -sk[(i, j)];
        }
    }

    // d(p_c) / d(world_pt)  — 3×3 = R
    let mut dpc_dp = DMatrix::<f64>::zeros(3, 3);
    for i in 0..3 {
        for j in 0..3 {
            dpc_dp[(i, j)] = pose.rotation[(i, j)];
        }
    }

    let j_pose = &dpix_dpc * &dpc_dpose; // 2×6
    let j_point = &dpix_dpc * &dpc_dp; // 2×3
    Some((j_pose, j_point))
}

/// Stack a set of observations into a full (weighted) Jacobian and residual.
///
/// Purely a convenience used by `schur` and the integration tests. Each
/// observation is the projection of point `point_idx` into camera `cam_idx`.
pub struct Observation {
    pub cam_idx: usize,
    pub point_idx: usize,
    pub measured: Vector2<f64>,
    /// 2×2 inverse-covariance weight (whitening) matrix. Use identity for
    /// isotropic unit-variance observations.
    pub weight_sqrt: DMatrix<f64>,
}

impl Observation {
    pub fn isotropic(cam_idx: usize, point_idx: usize, measured: Vector2<f64>) -> Self {
        let mut w = DMatrix::<f64>::zeros(2, 2);
        w[(0, 0)] = 1.0;
        w[(1, 1)] = 1.0;
        Self {
            cam_idx,
            point_idx,
            measured,
            weight_sqrt: w,
        }
    }
}

/// Residual vector for the full observation set, concatenated as `[r_1; r_2; ...]`
/// with whitening applied. Convenience only.
pub fn residuals(
    poses: &[SE3],
    intrinsics: &[PinholeIntrinsics],
    world_pts: &[Vector3<f64>],
    obs: &[Observation],
) -> DVector<f64> {
    let mut r = DVector::<f64>::zeros(2 * obs.len());
    for (i, o) in obs.iter().enumerate() {
        let predicted = project(&poses[o.cam_idx], &intrinsics[o.cam_idx], &world_pts[o.point_idx]);
        let diff = match predicted {
            Some(p) => p - o.measured,
            None => Vector2::zeros(), // behind-camera: zero residual for now
        };
        // Apply whitening sqrt(W)
        let w = &o.weight_sqrt;
        r[2 * i] = w[(0, 0)] * diff.x + w[(0, 1)] * diff.y;
        r[2 * i + 1] = w[(1, 0)] * diff.x + w[(1, 1)] * diff.y;
    }
    r
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_relative_eq;

    #[test]
    fn project_identity_pose() {
        let pose = SE3::identity();
        let k = PinholeIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let p = Vector3::new(0.0, 0.0, 5.0);
        let pix = project(&pose, &k, &p).unwrap();
        assert_relative_eq!(pix.x, 320.0, epsilon = 1e-9);
        assert_relative_eq!(pix.y, 240.0, epsilon = 1e-9);
    }

    #[test]
    fn project_behind_camera_returns_none() {
        let pose = SE3::identity();
        let k = PinholeIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let p = Vector3::new(0.0, 0.0, -1.0);
        assert!(project(&pose, &k, &p).is_none());
    }

    #[test]
    fn jacobian_matches_finite_difference() {
        let pose = SE3::identity();
        let k = PinholeIntrinsics::new(500.0, 500.0, 320.0, 240.0);
        let p = Vector3::new(0.1, -0.2, 5.0);

        let (j_pose, j_point) = projection_jacobian(&pose, &k, &p).unwrap();

        // Finite-difference wrt point coordinates
        let h = 1e-6;
        let base = project(&pose, &k, &p).unwrap();
        for dim in 0..3 {
            let mut p_perturbed = p;
            p_perturbed[dim] += h;
            let pred = project(&pose, &k, &p_perturbed).unwrap();
            let fd = (pred - base) / h;
            assert_relative_eq!(j_point[(0, dim)], fd.x, epsilon = 1e-4);
            assert_relative_eq!(j_point[(1, dim)], fd.y, epsilon = 1e-4);
        }

        // Finite-difference wrt translation component of pose
        for dim in 0..3 {
            let mut pose_p = pose.clone();
            pose_p.translation[dim] += h;
            let pred = project(&pose_p, &k, &p).unwrap();
            let fd = (pred - base) / h;
            assert_relative_eq!(j_pose[(0, dim)], fd.x, epsilon = 1e-4);
            assert_relative_eq!(j_pose[(1, dim)], fd.y, epsilon = 1e-4);
        }
    }
}
