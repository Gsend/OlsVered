# OlsVered / olssm_vision — Session Handoff

**Last worked on:** 2026-04-21
**User:** Gilad (giladsnd@gmail.com)
**Repo:** `C:\Users\nat79\OlsVered`  (workspace-mounted as `OlsVered`)
**Active branch:** `feature/olssm-vision`

---

## One-paragraph resume

We built Phase 0 + Phase 1 of a new `olssm_vision` crate that layers SLAM / AR / SfM primitives on top of the existing `olssm` closed-form OLS kernels (from Vered Madar's research). It's a standalone sibling crate with a path-dependency on root `olssm` — zero changes to root `Cargo.toml`, `src/`, `optimizer/`, or `python/olssm/`. Two commits are in on `feature/olssm-vision`. Nothing has been `cargo test`-validated yet because the sandbox has no Rust toolchain, and Gilad's local Windows machine also doesn't have Rust installed yet. **Immediate next step: install Rust via rustup, then `cd olssm_vision && cargo test`.**

---

## Repo state

### Commits on `feature/olssm-vision`

```
47dce96  Add olssm_vision crate — Phase 0 + Phase 1 of SLAM/AR/SfM module
486566b  Check git state and create branch olssm_vision (Cargo.toml + error.rs + geometry.rs)
```

### Files created this session (all committed)

```
olssm_vision/
  Cargo.toml                     # path dep on root olssm; feature "python" gates PyO3
  README.md                      # module overview + scope
  src/
    error.rs                     # VisionError + From<olssm::OlsSMError>
    geometry.rs                  # SE(3), PinholeIntrinsics, project, 2x6+2x3 Jacobian
    schur.rs                     # build_schur, SchurSystem, backsubstitute_points
    semi_partial.rs              # whitened Cholesky, pivot profile, near-singular, IC pattern
    selected_inverse.rs          # Takahashi column / diagonal / block
    lib.rs                       # PyO3 bindings exposed as olssm_vision._rust
  tests/
    test_end_to_end.rs           # 3-cam x 4-pt scene: Schur, Takahashi, gauge detection

python/olssm_vision/
  __init__.py                    # re-exports Rust extension with numpy fallback
  pyproject.toml                 # maturin config -> olssm_vision._rust
  diagnostics.py                 # DegeneracyReport, analyse_information, leverage
  slam.py                        # CovarianceOracle (marginal / cross / point)
  ba.py                          # solve_ba_step (one damped LM inner step)

documentation/
  vision_module_plan.md          # 4-phase plan; Phase 0-1 done, Phase 2-4 TODO
```

### What was NOT touched (by design)

- root `Cargo.toml`
- `src/` (root crate — contains existing `algorithms.rs`, `lib.rs`)
- `optimizer/`
- `python/olssm/` (existing Python package)
- `benchmark/` (there are unrelated pre-existing dirty changes in `benchmark/results/` — leave alone)

---

## Immediate next step — run the Rust tests

Rust is not installed on `C:\Users\nat79`. Install via either:

```powershell
winget install Rustlang.Rustup
# then close + reopen PowerShell
rustup default stable
```

or download `rustup-init.exe` from https://rustup.rs. May also need MSVC C++ build tools from https://visualstudio.microsoft.com/downloads/ → "Desktop development with C++".

Then:

```powershell
cd C:\Users\nat79\OlsVered\olssm_vision
cargo test --no-default-features     # skip the python feature (not needed for Rust tests)
```

Expected: ~10 test functions across the unit tests and integration test pass. First build is slow (downloads nalgebra, faer, approx, thiserror); subsequent runs cached.

### Expected test names

- `geometry::tests::project_identity_pose`
- `geometry::tests::project_behind_camera_returns_none`
- `geometry::tests::jacobian_matches_finite_difference`
- `schur::tests::invert_3x3_of_identity_block`
- `schur::tests::schur_matches_direct_inverse_on_toy_problem`
- `semi_partial::tests::*`  (whitened Cholesky on known SPD, pivot_profile, detect_near_singular)
- `selected_inverse::tests::*`  (column / diagonal / block vs direct inverse)
- `tests/test_end_to_end.rs`:
  - `integration_schur_matches_full_solve`
  - `integration_covariance_matches_direct_inverse`
  - `integration_gauge_detection_on_fixed_camera_pair`

### Known potential gotcha

If nalgebra 0.33's view-matrix multiplication trips the borrow checker in `schur.rs` around the line `let product = b_slab * &vinv_dyn;`, the one-line fix is:

```rust
let b_slab = b.columns(base, 3).into_owned();
```

This was the deferred optional safety fix; I left it as-is because the current form should compile on recent nalgebra. If it doesn't, apply this and re-run.

### After cargo test passes

```powershell
cd python\olssm_vision
pip install maturin
maturin develop --features python
pytest -v    # if any pytest files exist; otherwise just try `python -c "import olssm_vision; print(dir(olssm_vision))"`
```

---

## Open decisions (previously discussed; no action taken yet)

1. **Provisional patent + seed funding** — Gilad asked "is it worth submitting for a provisional patent then raising seed money and developing this?" We gave a strategic answer leaning toward: provisional patent yes (cheap, 12-mo runway), seed money only after one deployed benchmark proves the claim. No draft patent app or pitch deck yet.

2. **Phase 2 — sparse backend + Madar native Cholesky kernel** — All current paths are dense. Planned swap: `faer-sparse` or `sprs` for the Schur `S` matrix, and a native implementation of Madar's closed-form Modified Cholesky (currently using `nalgebra::Cholesky`). The API is designed to be swap-in-compatible.

3. **Phase 3 — robust BA + 3DGS export** — LM outer loop with Huber / Tukey reweighting, exporter for 3D Gaussian Splatting consumers.

4. **Phase 4 — benchmarks** — vs GTSAM, Ceres, COLMAP. Target datasets: KITTI, ScanNet, EuRoC.

---

## Prior business/technical analysis (earlier in session, before code)

Key claims from the analysis, for context when future discussions reference them:

- **Target benefit**: Madar's Modified Cholesky offers ~10x memory reduction and O(n²) selected-inverse via Takahashi for BA covariance recovery — plugs into SLAM (loop-closure uncertainty), AR (anchor stability), SfM (dense reconstruction quality gate).
- **Addressable markets discussed**: photogrammetry vendors (Pix4D, Bentley, Agisoft), drone/robotics (Skydio, DJI-Enterprise, Shield AI), AR/VR platforms (Meta, Apple, Snap, Niantic), NeRF / 3DGS pipelines (Luma, Polycam, Nerfstudio), defense/geospatial (Maxar, BlackSky, Palantir).
- **Competitive landscape**: classical BA (Ceres, g2o, GTSAM) — mature but covariance recovery is weak; neural methods (NeRF, 3DGS, DUSt3R, MASt3R, VGGT) — compete on end-task but not on uncertainty quantification. Madar's angle is *calibrated uncertainty* at numerical-linear-algebra speed, not learned end-to-end.
- **Monetary value estimates given** (order-of-magnitude, not hard numbers): memory reduction translates to deployable-on-edge-device capability worth low-single-digit $M ARR per large customer; gauge detection + preconditioner speedup could cut BA iteration time 2–5x, worth similar per customer; aggregate TAM for uncertainty-aware SLAM/SfM infra was framed in the $100M–$1B range over 5 years if execution lands.

---

## Key technical decisions

- **Isolation strategy**: standalone crate `olssm_vision/` with `olssm = { path = ".." }` in its Cargo.toml, rather than converting root into a workspace. Keeps blast radius zero on root crate.
- **Feature-gated PyO3**: Rust tests don't need Python headers; use `cargo test --no-default-features`.
- **Python fallbacks everywhere**: every Python module in `python/olssm_vision/` works before `maturin develop` by falling back to pure numpy. The Rust extension is imported optionally.
- **PyO3 module name**: `#[pymodule] fn _rust` in `lib.rs` matches `module-name = "olssm_vision._rust"` in `pyproject.toml`. Do not rename without updating both.
- **Libraries pinned** (from root repo): `nalgebra 0.33`, `faer 0.19`, `PyO3 0.21`, `thiserror 1.0`, `approx 0.5`.

---

## How to resume in a future Claude session

Paste the user prompt:

> I want to continue work on olssm_vision. Please read `SESSION_HANDOFF.md` in the repo root to catch up on state, then [pick one]:
> (a) help me get `cargo test` passing on my Windows machine
> (b) start Phase 2 (sparse backend + native Madar Cholesky kernel)
> (c) draft the provisional patent application
> (d) draft a seed fundraising deck
> (e) [other]

The full transcript of this session is at:
`/sessions/intelligent-loving-knuth/mnt/.claude/projects/-sessions-intelligent-loving-knuth/cf35f38e-6e01-476e-aa5c-59f5ccce9f17.jsonl`
— but this handoff doc should be sufficient for pickup without reading the transcript.
