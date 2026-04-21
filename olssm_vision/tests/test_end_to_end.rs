//! End-to-end integration test: synthetic 3-camera / 4-point BA scene,
//! Schur elimination, solve, covariance recovery, and degeneracy detection.

use approx::assert_relative_eq;
use nalgebra::{DMatrix, DVector, Matrix3, Vector2, Vector3};
use olssm_vision::{geometry, schur, selected_inverse, semi_partial};

fn make_toy_scene() -> (Vec<geometry::SE3>, Vec<geometry::PinholeIntrinsics>, Vec<Vector3<f64>>) {
    let mut poses = Vec::new();
    // Camera 0: at origin, looking along +Z
    poses.push(geometry::SE3::identity());
    // Camera 1: shifted +x by 0.5
    let mut c1 = geometry::SE3::identity();
    c1.translation = Vector3::new(-0.5, 0.0, 0.0);
    poses.push(c1);
    // Camera 2: shifted -x by 0.5
    let mut c2 = geometry::SE3::identity();
    c2.translation = Vector3::new(0.5, 0.0, 0.0);
    poses.push(c2);

    let k = geometry::PinholeIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let intrinsics = vec![k, k, k];

    // Four 3-D points spread through the scene
    let points = vec![
        Vector3::new(0.1, 0.0, 5.0),
        Vector3::new(-0.2, 0.1, 4.0),
        Vector3::new(0.3, -0.1, 6.0),
        Vector3::new(0.0, 0.25, 4.5),
    ];

    (poses, intrinsics, points)
}

/// Build a fully populated block-Hessian `H = Jᵀ J + λI` and its RHS from the
/// projections of every point into every camera. Cameras first, then points.
fn build_ba_system(
    poses: &[geometry::SE3],
    intrinsics: &[geometry::PinholeIntrinsics],
    points: &[Vector3<f64>],
    damping: f64,
) -> (DMatrix<f64>, DMatrix<f64>, DMatrix<f64>, DVector<f64>, DVector<f64>) {
    let nc = poses.len();
    let np_ = points.len();
    let mut u = DMatrix::<f64>::zeros(6 * nc, 6 * nc);
    let mut v = DMatrix::<f64>::zeros(3 * np_, 3 * np_);
    let mut b = DMatrix::<f64>::zeros(6 * nc, 3 * np_);

    for ci in 0..nc {
        for pi in 0..np_ {
            let jac = geometry::projection_jacobian(&poses[ci], &intrinsics[ci], &points[pi]);
            if jac.is_none() {
                continue;
            }
            let (j_pose, j_point) = jac.unwrap();
            // U_ci += j_pose^T j_pose
            let uu = j_pose.transpose() * &j_pose;
            for r in 0..6 {
                for c in 0..6 {
                    u[(6 * ci + r, 6 * ci + c)] += uu[(r, c)];
                }
            }
            // V_pi += j_point^T j_point
            let vv = j_point.transpose() * &j_point;
            for r in 0..3 {
                for c in 0..3 {
                    v[(3 * pi + r, 3 * pi + c)] += vv[(r, c)];
                }
            }
            // B_{ci, pi} += j_pose^T j_point
            let bb = j_pose.transpose() * &j_point;
            for r in 0..6 {
                for c in 0..3 {
                    b[(6 * ci + r, 3 * pi + c)] += bb[(r, c)];
                }
            }
        }
    }

    // Add damping to U and V diagonals.
    for i in 0..6 * nc {
        u[(i, i)] += damping;
    }
    for i in 0..3 * np_ {
        v[(i, i)] += damping;
    }

    // Use arbitrary but deterministic RHS vectors for the linear solve test.
    let g_c = DVector::<f64>::from_fn(6 * nc, |i, _| (i as f64 + 1.0).cos());
    let g_p = DVector::<f64>::from_fn(3 * np_, |i, _| (i as f64 + 0.5).sin());

    (u, v, b, g_c, g_p)
}

#[test]
fn integration_schur_matches_full_solve() {
    let (poses, intrinsics, points) = make_toy_scene();
    let (u, v, b, g_c, g_p) = build_ba_system(&poses, &intrinsics, &points, 1.0);

    let nc = 6 * poses.len();
    let np_ = 3 * points.len();

    let schur_sys = schur::build_schur(&u, &v, &b, &g_c, &g_p, 0.0).unwrap();

    // Full reference solve
    let mut h = DMatrix::<f64>::zeros(nc + np_, nc + np_);
    h.view_mut((0, 0), (nc, nc)).copy_from(&u);
    h.view_mut((nc, nc), (np_, np_)).copy_from(&v);
    h.view_mut((0, nc), (nc, np_)).copy_from(&b);
    h.view_mut((nc, 0), (np_, nc)).copy_from(&b.transpose());

    let mut g = DVector::<f64>::zeros(nc + np_);
    g.view_mut((0, 0), (nc, 1)).copy_from(&g_c);
    g.view_mut((nc, 0), (np_, 1)).copy_from(&g_p);

    let delta_full = h.lu().solve(&g).unwrap();
    let delta_c = schur_sys.s.clone().lu().solve(&schur_sys.g_tilde).unwrap();
    let delta_p = schur_sys.backsubstitute_points(&delta_c);

    for i in 0..nc {
        assert_relative_eq!(delta_c[i], delta_full[i], epsilon = 1e-6);
    }
    for i in 0..np_ {
        assert_relative_eq!(delta_p[i], delta_full[nc + i], epsilon = 1e-6);
    }
}

#[test]
fn integration_covariance_matches_direct_inverse() {
    let (poses, intrinsics, points) = make_toy_scene();
    let (u, v, b, g_c, g_p) = build_ba_system(&poses, &intrinsics, &points, 1.0);
    let schur_sys = schur::build_schur(&u, &v, &b, &g_c, &g_p, 0.0).unwrap();

    // Covariance of the reduced camera system
    let l = nalgebra::Cholesky::new(schur_sys.s.clone()).unwrap().l();
    let inv_ref = schur_sys.s.clone().try_inverse().unwrap();

    // Full diagonal via Takahashi
    let diag = selected_inverse::diagonal(&l).unwrap();
    for i in 0..schur_sys.s.nrows() {
        assert_relative_eq!(diag[i], inv_ref[(i, i)], epsilon = 1e-8);
    }

    // 6×6 block for camera 0 (parameters 0..6)
    let blk = selected_inverse::block(&l, &[0, 1, 2, 3, 4, 5]).unwrap();
    for i in 0..6 {
        for j in 0..6 {
            assert_relative_eq!(blk[(i, j)], inv_ref[(i, j)], epsilon = 1e-8);
        }
    }
}

#[test]
fn integration_gauge_detection_on_fixed_camera_pair() {
    // If two cameras are at the same position with identical pose, they
    // contribute duplicate information to many parameters — creating a
    // near-singular direction. We fabricate that by stacking identical
    // observations into U.
    let (_, _, points) = make_toy_scene();
    let poses = vec![geometry::SE3::identity(); 2];
    let k = geometry::PinholeIntrinsics::new(500.0, 500.0, 320.0, 240.0);
    let intrinsics = vec![k; 2];

    let (u, v, b, g_c, g_p) = build_ba_system(&poses, &intrinsics, &points, 0.1);
    let schur_sys = schur::build_schur(&u, &v, &b, &g_c, &g_p, 0.0).unwrap();

    let (lw, _d) = semi_partial::whitened_cholesky(&schur_sys.s).unwrap();
    let weak = semi_partial::detect_near_singular(&lw, 0.05);

    // We expect at least one near-singular direction in this degenerate pair.
    assert!(
        !weak.is_empty(),
        "Expected gauge/degeneracy detection with duplicate cameras; got none"
    );
}
