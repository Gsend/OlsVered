# `olssm_vision` — design plan

## Motivation

The `olssm` crate implements Madar & Batista's closed-form OLS algorithms
(LU / unnormalised Gram-Schmidt / weighted generalised inverse) and exposes
them as a Rust library with Python and C bindings. The underlying
mathematical primitives — direct-formulation Cholesky factors, semi-partial
correlation coefficients, and inversion-free least-squares solves — are
immediately applicable to bundle adjustment and SLAM covariance recovery,
but the core crate is intentionally scope-limited to dense `(X, y)` OLS
problems.

`olssm_vision` is a sibling crate that adapts these primitives for:

1. **Covariance-on-demand** for online SLAM / AR, via a selected-inverse
   (Takahashi) recursion over the Cholesky factor. Column, diagonal, and
   small-block queries execute in `O(n²)` / `O(k · n²)` flops without
   materialising the full dense inverse.

2. **Gauge / degeneracy diagnostics** for capture-time feedback to users
   of drone / phone photogrammetry pipelines. Diagonal pivots of the
   whitened Cholesky factor are interpreted as semi-partial correlation
   magnitudes; near-zero pivots flag the 7-DoF similarity gauge and any
   additional unobservable directions (planar scenes, pure rotation, etc.).

3. **Weighted / heteroskedastic BA** via the existing
   `weighted_generalized_inverse` primitive, enabling per-keypoint
   uncertainty from learned feature matchers to be carried through the
   inner LM solve at negligible extra cost.

4. **Leverage** (hat-matrix diagonal) for robust BA, derived from the
   existing `simplified_gram_schmidt` primitive.

## Phase summary

| Phase | Scope                                                                   | Status |
|-------|-------------------------------------------------------------------------|--------|
| 0     | Workspace / crate scaffolding; dependencies; Python package stubs       | Done   |
| 1     | Dense Schur + whitened Cholesky + selected-inverse + gauge detector     | Done   |
| 2     | Sparse backend; Madar closed-form Cholesky kernel; COLMAP I/O           | TODO   |
| 3     | Leverage-based robust BA; 3DGS / NeRF covariance export; active mapping | TODO   |
| 4     | Benchmarks vs GTSAM / Ceres; tech report; patent-adjacent write-up      | TODO   |

## Phase 0 — scaffolding (done)

- New crate `olssm_vision/` with `Cargo.toml` declaring a **path dependency**
  on the root `olssm` crate. The root `Cargo.toml` is unchanged — `olssm`
  still builds and tests stand-alone.
- Python companion package under `python/olssm_vision/`, built via maturin
  as a separate extension named `olssm_vision._rust`. Its `pyproject.toml`
  points at the sibling Rust crate.
- Reserved directories `vision_benchmark/` and `vision_tests/` for Phase 4.

## Phase 1 — core primitives (done)

### Modules

- `error.rs` — `VisionError` enum, `From<olssm::algorithms::OlsSMError>`.
- `geometry.rs` — `SE3`, `PinholeIntrinsics`, `project`, analytic
  `projection_jacobian`, `Observation`, `residuals`. Finite-difference test
  confirms the analytic Jacobian. Dynamic matrix storage for API consistency
  with `olssm`; small-size specialisations left as a performance follow-up.
- `schur.rs` — `build_schur(U, V, B, g_c, g_p, damping)` returns a
  `SchurSystem` carrying `S`, `g̃`, the cached `V⁻¹` blocks and `B`, and a
  `backsubstitute_points(δ_c)` method. Unit test verifies the reduced
  solution matches a full direct solve on a toy 12-camera / 9-point system.
- `semi_partial.rs` — `whitened_cholesky(A)` returns `(L̃, d)` where
  `L̃` is Cholesky of `D⁻¹ᐟ² A D⁻¹ᐟ²`. Helpers `pivot_profile`,
  `detect_near_singular`, `sparsity_template`. Unit tests confirm
  reconstruction and gauge detection on a synthetic near-degenerate matrix.
- `selected_inverse.rs` — `column(L, j)`, `diagonal(L)` (Takahashi
  backward recursion), `block(L, indices)`. Unit tests check agreement with
  direct inverse on 7-to-10-dim SPD matrices.
- `lib.rs` — PyO3 bindings under the `python` feature, exposing all of the
  above as `olssm_vision._rust`.

### Integration test

`tests/test_end_to_end.rs` constructs a 3-camera / 4-point synthetic scene,
builds the full BA Hessian from analytic projection Jacobians, reduces via
Schur, and asserts:

1. Schur-reduced solution matches full direct solve (rel tol `1e-6`).
2. Takahashi diagonal and 6×6 camera block match direct inverse (rel tol `1e-8`).
3. Duplicate-camera degenerate case produces at least one near-singular
   column via `semi_partial::detect_near_singular`.

### Python API

- `olssm_vision.diagnostics.analyse_information(S)` → `DegeneracyReport`.
- `olssm_vision.diagnostics.leverage(J)` → per-row hat matrix diagonal.
- `olssm_vision.slam.CovarianceOracle.from_information(S)` → oracle with
  `parameter_variance(i)`, `cross_covariance(i, j)`, `point_covariance(dof)`,
  `all_marginal_variances()`.
- `olssm_vision.ba.solve_ba_step(U, V, B, g_c, g_p, damping)` → `BAStep`.

All Python paths have pure-numpy fallbacks usable before the Rust
extension is built — development and CI don't block on maturin.

## Phase 2 — sparsity and Madar kernel (TODO)

- Add `faer-sparse` dependency; retrofit `schur::build_schur` with a sparse
  path exploiting the co-visibility graph. Expected memory reduction 3–10×
  on medium scenes where `B` is sparse.
- Replace the `nalgebra::Cholesky` call in `semi_partial::whitened_cholesky`
  with a direct evaluation that reuses `olssm::algorithms::modified_cholesky`
  and then de-augments the `[X | y]` convention. This is the "Madar form"
  of the factor and is the step that unlocks column-streaming memory
  savings.
- AMD / METIS fill-reducing ordering. The semi-partial-correlation
  interpretation becomes ordering-dependent; the gauge detector must
  report in post-permutation indices and provide a mapping back to the
  user's parameter order.
- COLMAP binary-format reader / writer (`cameras.bin`, `images.bin`,
  `points3D.bin`) under `integration/colmap.rs` + `integration/colmap.py`.

## Phase 3 — robust BA and downstream consumers (TODO)

- Leverage-based reweighting hook in the LM outer loop (Python).
- Per-point covariance export for 3DGS trainers in the `.npz` sidecar
  format used by current Nerfstudio branches.
- Next-best-view helper: given a candidate SE(3) pose, predict the
  expected reduction in the largest ellipsoid axis.

## Phase 4 — benchmarks (TODO)

- `vision_benchmark/bench_covariance.py` — latency + accuracy vs
  GTSAM `Marginals::marginalCovariance` on TUM-RGBD.
- `vision_benchmark/bench_ba_solver.py` — accuracy + wall-clock vs Ceres
  Schur-Cholesky on ETH3D indoor scenes.
- `vision_benchmark/bench_degeneracy.py` — planar-scene and pure-rotation
  synthetic cases.
- Patent-adjacent write-up extending `documentation/patent_application.docx`
  with claims on the SLAM / AR applications.

## Open questions to resolve before Phase 2

1. Sparse backend: `faer-sparse` (preferred; stays in-ecosystem) vs `sprs`
   (more mature, GPL-licence mix with SuiteSparse).
2. Pivoting in the Madar LU on near-singular `S`: LM damping only, or
   add partial pivoting for stability at the cost of closed-form elegance.
3. Python package shape: separate `olssm_vision` on PyPI, or `olssm[vision]`
   extras install.
4. Patent scope: confirm whether covariance-recovery and degeneracy-detection
   uses are within scope of the existing application or require a fresh
   provisional filing.
