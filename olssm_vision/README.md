# olssm_vision

SLAM / AR / SfM applications of the `olssm` closed-form OLS kernels.

This crate layers four capabilities on top of the root `olssm` crate without
modifying it:

| Module             | Purpose                                                                 |
|--------------------|-------------------------------------------------------------------------|
| `geometry`         | SE(3) poses, pinhole intrinsics, projection + analytic Jacobians        |
| `schur`            | Bundle-adjust reduced-camera system via Schur complement                |
| `semi_partial`     | Whitened Cholesky factor, semi-partial correlations, gauge diagnostics  |
| `selected_inverse` | On-demand `A⁻¹` columns / diagonal / blocks (Takahashi recursion)       |

Plus a companion Python package in `../python/olssm_vision/` exposing:

- `diagnostics.analyse_information(S)` — gauge / degeneracy report for a BA
  information matrix.
- `diagnostics.leverage(J)` — per-observation hat-matrix leverage.
- `slam.CovarianceOracle` — cached-Cholesky oracle serving per-parameter
  variance, cross-covariance, and 3×3 landmark blocks in one call each.
- `ba.solve_ba_step(U, V, B, g_c, g_p, damping)` — one LM inner step on a
  pre-linearised BA problem.

## Relationship to `olssm`

`olssm_vision` depends on `olssm` by **path** (`olssm = { path = ".." }`).
It does not modify any file inside the `olssm` crate or inside `optimizer/`.
All additions live in:

```
olssm_vision/              # this crate
python/olssm_vision/       # companion Python package
vision_benchmark/          # reserved for Phase 4 benchmarks
vision_tests/              # reserved for Python integration tests
```

## Build

```bash
# Rust only
cd olssm_vision
cargo build
cargo test

# Python extension (from repo root)
cd python/olssm_vision
maturin develop --features python
pytest -v
```

## Scope of this first slice

What is implemented now (Phase 0 + core of Phase 1):

- Dense Schur complement with per-point 3×3 `V⁻¹` inversion
- Dense whitened Cholesky + semi-partial correlation access
- Dense Takahashi-recursion diagonal, column, and block selected-inverse
- Synthetic-scene integration test (3 cameras × 4 points) that compares
  against a full direct solve and a full direct inverse

What is **not** yet implemented and is cleanly deferred:

- Sparse matrix backend (`faer-sparse` / `sprs`). All current paths are
  dense and scale to ~10⁴ parameter dimensions.
- Native Madar closed-form Cholesky (`modified_cholesky` from `olssm`).
  Phase 1 uses `nalgebra::Cholesky` for now; swapping in the Madar kernel
  is a Phase 2 task that preserves the external API.
- COLMAP / GTSAM / Ceres data interop (`integration/`).
- LM outer loop with robust-norm reweighting (`ba.py` currently does one
  damped step only).
- 3DGS / NeRF point-ellipsoid export.

See the top-level design plan (conversation transcript in
`documentation/vision_module_plan.md`) for phase-by-phase scope.

## License

MIT, matching the root crate.
