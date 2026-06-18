# olssm — Publishing Plan (Rust crate + Python wrapper)

A handoff document for building and publishing `olssm`: a faithful Rust
implementation of the Madar/Batista closed-form OLS algorithms, with Python
bindings. Written to be self-contained — a fresh session (Claude Cowork) should
be able to execute it without prior context.

---

## 0. Context & goal

**Source paper:** Vered Senderovich Madar & Sandra L. Batista, *"Solving The
Ordinary Least Squares in Closed Form, Without Inversion or Normalization"*,
arXiv:2301.01854 (v2, Dec 2023).

**What the paper offers (the algorithms to implement):**
- Closed-forms for OLS coefficients via the LU factorization of the Gram matrix,
  where the upper-triangular factor is **NOT unit-upper-triangular** (this is the
  paper's key distinction — see §3 correctness note).
- Simplified Gram-Schmidt Orthogonalization (SGSO) **without normalization** —
  avoids all square roots, uses only dot products.
- A closed-form for the generalized inverse.
- A closed-form for **weighted** linear regression coefficients.
- A simplification of the Frisch-Waugh-Lovell computation.

**Why publish (all three goals must be served):**
1. *Adoption* — a usable library. The audience is Python statistics/ML users, so
   the Python wrapper is essential, not optional.
2. *Citable artifact* — first faithful implementation of these closed-forms in
   Rust; README cites arXiv:2301.01854. Requires correctness fidelity (§3).
3. *Portfolio / job-search signal* — clean Rust + numerical LA + PyO3, relevant
   to frontier ML / Swiss CV-ML roles. Requires polish (tests, docs, CI).

**Explicit non-goals (decided):**
- **No GPU implementation.** The closed-forms are inherently sequential
  (back-substitution, non-normalized GSO inner loops); GPU loses to CPU at the
  small `p` where these methods matter. Generic GPU LU/eigh already exists in
  cuSOLVER/PyTorch and is not novel.
- **No K-FAC / faer optimizer code in this repo.** The `lu_*_gram`, `eigh_*`,
  `randomized_eigh`, `apply_kfac_*` functions belong to the separate OlsVered
  optimizer project. Keep them out of `olssm` — different audience, different
  novelty story, ties to unpublished research. (A future `olsvered` repo can
  depend on `olssm` if useful.)

**License:** Dual MIT OR Apache-2.0 (Rust ecosystem convention). Include both
`LICENSE-MIT` and `LICENSE-APACHE`; set `license = "MIT OR Apache-2.0"` in
`Cargo.toml`.

**Ecosystem integration boundary (decided — do not cross the streams):**
- `olssm` integrates with **numpy only**, never torch. These are statistics/
  econometrics methods; the audience uses numpy arrays, sklearn pipelines, and
  statsmodels-style workflows — not torch training loops. numpy in/out (the PyO3
  bindings already do this) plus a scikit-learn-compatible estimator interface is
  the complete integration story.
- **Do NOT add a torch dependency to `olssm`.** It would add a massive, slow
  dependency that ~95% of users don't want, to serve a use case that belongs to
  the separate optimizer project. Keeping `olssm` lightweight (numpy-only) is a
  hard requirement.
- numpy-only does **not** lock out torch users: numpy↔torch interop is cheap
  (`torch.from_numpy`, `.numpy()`, DLPack/array-interface), so a rare torch-context
  user converts at the boundary in one line. `olssm` is torch-*compatible* without
  torch *integration*.
- **torch belongs to the separate OlsVered optimizer project**, where a K-FAC
  preconditioner is only meaningful inside a training loop and its natural home is
  a `torch.optim.Optimizer` subclass. That is out of scope here. If OlsVered ever
  needs the OLS core, *it* depends on `olssm` — never the reverse.

| Artifact | Integrates with | Why |
|---|---|---|
| `olssm` (this repo, Algs 1–3) | **numpy** (+ sklearn API) | statistics audience; array in/out |
| OlsVered (separate repo, K-FAC) | **torch** (`Optimizer` subclass) | only meaningful inside training |

---

## 1. Scope: what goes in the crate

Port **only** these from the existing Rust source (the pure-nalgebra core):

| Paper algorithm | Function(s) | Notes |
|---|---|---|
| Alg 1: Gram-matrix LU closed-form | `gram_lu_factor` + `back_substitute` | RENAME from `modified_cholesky`; fix normalization (§3) |
| Combined OLS solve | `solve_ols` | convenience: factor + back-substitute |
| Alg 2: SGSO (no normalization) | `simplified_gram_schmidt` | square-root-free; keep as-is, add tests |
| Alg 3: weighted generalized inverse | `weighted_generalized_inverse` | `(XᵀWX)⁻¹XᵀW` via LU solve; keep, add tests |
| (derived) generalized inverse | `generalized_inverse` | NEW: expose the unweighted `(XᵀX)⁻¹Xᵀ` closed-form the paper derives |

Drop everything from "Algorithm 4" onward (faer, K-FAC, eigh, randomized SVD).

**Python-side adoption feature (high leverage — include in v0.1 or fast-follow
v0.2):** a scikit-learn-compatible estimator wrapping the bindings — an
`OlsRegressor(BaseEstimator, RegressorMixin)` with `fit(X, y)`, `predict(X)`, and
`coef_` / `intercept_` attributes. This is the difference between "another OLS
function" (ignorable) and "a drop-in sklearn regressor" (adoptable into pipelines,
cross-validation, grid search). Lives in `python/olssm/` as pure Python over the
compiled core. numpy-only, no torch.

---

## 2. Repository layout (maturin mixed Rust/Python project)

```
olssm/
├── Cargo.toml              # crate metadata, deps, license
├── pyproject.toml          # maturin backend, project metadata for PyPI
├── README.md               # overview, citation, examples, benchmarks
├── LICENSE-MIT
├── LICENSE-APACHE
├── CHANGELOG.md
├── .github/workflows/
│   ├── ci.yml              # cargo test + clippy + fmt + python tests
│   └── release.yml         # maturin build wheels + publish (manual trigger)
├── src/
│   ├── lib.rs              # pub use of the math module; crate docs
│   ├── ols.rs             # Algorithms 1–3 + derived inverse (pure Rust core)
│   ├── error.rs           # OlsError enum (thiserror)
│   └── python.rs          # PyO3 bindings (feature-gated: "python")
├── python/
│   └── olssm/
│       ├── __init__.py    # re-export, version, numpy-friendly docstrings
│       └── _typing.pyi    # type stubs for the compiled functions
├── tests/
│   ├── correctness.rs     # Rust: compare β vs nalgebra normal-equations
│   ├── properties.rs      # Rust: proptest random systems, edge cases
│   └── test_python.py     # Python: compare vs numpy.linalg.lstsq
├── benches/
│   └── ols_bench.rs       # criterion: olssm vs nalgebra QR vs normal-eq
└── examples/
    └── basic.rs           # minimal usage example
```

**Cargo.toml essentials:**
```toml
[package]
name = "olssm"
version = "0.1.0"
edition = "2021"
license = "MIT OR Apache-2.0"
description = "Closed-form OLS without matrix inversion or normalization (Madar & Batista 2023)"
repository = "https://github.com/<user>/olssm"
keywords = ["least-squares", "regression", "linear-algebra", "ols", "statistics"]
categories = ["science", "mathematics"]

[lib]
name = "olssm"
crate-type = ["cdylib", "rlib"]   # cdylib for Python, rlib for Rust consumers

[dependencies]
nalgebra = "0.33"
thiserror = "2"
numpy = { version = "0.22", optional = true }
pyo3 = { version = "0.22", features = ["extension-module"], optional = true }

[features]
default = []
python = ["dep:pyo3", "dep:numpy"]

[dev-dependencies]
approx = "0.5"
proptest = "1"
criterion = "0.5"

[[bench]]
name = "ols_bench"
harness = false
```
*(Pin exact versions at build time — check latest nalgebra/pyo3/numpy compatibility;
pyo3 and the `numpy` crate versions must match.)*

**pyproject.toml essentials:**
```toml
[build-system]
requires = ["maturin>=1.5,<2"]
build-backend = "maturin"

[project]
name = "olssm"
version = "0.1.0"
description = "Closed-form OLS without inversion or normalization"
readme = "README.md"
license = { text = "MIT OR Apache-2.0" }
requires-python = ">=3.8"
dependencies = ["numpy>=1.21"]
classifiers = [
  "Programming Language :: Rust",
  "Topic :: Scientific/Engineering :: Mathematics",
]

[tool.maturin]
features = ["python"]
module-name = "olssm._olssm"
python-source = "python"
```

---

## 3. CORRECTNESS FIX — do this before anything else (highest priority)

**The bug:** the current `modified_cholesky` row-normalizes the LU upper factor to
force a unit diagonal:
```rust
// Row-normalise: C[i, :] = U[i, :] / U[i, i]  so diag(C) = 1.
```
This **contradicts the paper's central claim**, which is explicitly that the LU
factorization yields an upper-triangular factor that is *not* unit-upper-
triangular, and that GSO-without-normalization works *because* of this. The
"normalization" the paper's title rejects is precisely this kind of step.

**Required actions:**
1. **Re-derive against the paper.** Open arXiv:2301.01854 v2. Identify the exact
   closed-form for βᵢ as a linear combination of (a) the non-normalized GSO
   vectors and (b) the non-unit upper-triangular LU factor. Implement *that*
   formula, not a unit-diagonal back-substitution that happens to give the right
   answer.
2. **Rename** `modified_cholesky` → `gram_lu_factor` (the paper frames it as a
   generalization of Cholesky, but "modified Cholesky" is an unrelated existing
   term — avoid the name collision). Update all references.
3. **Verify the relationship** the paper establishes: LU factor of the Gram matrix
   ↔ SGSO vectors. Add a test asserting they're consistent (e.g. the diagonal of
   U equals the squared norms of the SGSO vectors, per the paper).
4. **Decide on the zero-pivot / exact-fit handling.** The current code special-
   cases the last (y-column) pivot being zero. Keep this behavior but document it
   precisely and test it (y exactly in column space of X → residual zero).

**Acceptance test for the fix:** for 1000 random `(X, y)` systems of varying
shape (tall, square-ish, varying condition number), `solve_ols(X, y)` must match
`numpy.linalg.lstsq(X, y)` (and nalgebra's normal-equation / SVD solve) to within
`1e-8` relative error on well-conditioned systems. If this fails, the
implementation is not citable — stop and fix.

---

## 4. Build steps (each ships something testable)

### Step A — Rust core crate
1. Create repo, `cargo init --lib`, add both license files, Cargo.toml above.
2. Move `OlsError` enum into `src/error.rs` (rename from `OlsSMError`; clean up
   variant names — `DimensionMismatch`, `ZeroPivot`, `WeightDimension`,
   `SingularMatrix`).
3. Port Algs 1–3 into `src/ols.rs`. **Apply the §3 fix.**
4. Add `generalized_inverse` (unweighted closed-form) for completeness.
5. Write `tests/correctness.rs` and `tests/properties.rs` (proptest).
6. `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt --check` all green.
7. `examples/basic.rs` runs.

**Ships:** a publishable Rust crate. Could `cargo publish` here already.

### Step B — Python bindings
1. Add `src/python.rs` behind the `python` feature. For each public function,
   write a `#[pyfunction]` that:
   - accepts `PyReadonlyArray2<f64>` / `PyReadonlyArray1<f64>` (numpy),
   - converts to `nalgebra::DMatrix`/`DVector` (zero-copy where possible, else a
     single copy),
   - calls the core function,
   - converts the result back to `PyArray` and returns it,
   - maps `OlsError` → a Python exception (`ValueError` / custom `OlsError`).
2. `#[pymodule]` named `_olssm` exporting the functions.
3. `python/olssm/__init__.py` re-exports from `_olssm`, adds numpy-friendly
   docstrings and a `__version__`.
4. Add `_typing.pyi` stubs so editors/type-checkers see signatures.
5. `maturin develop --features python`, then run `tests/test_python.py`
   comparing against `numpy.linalg.lstsq` and `statsmodels` OLS if available.

**Ships:** `import olssm; olssm.solve_ols(X, y)` works locally.

### Step C — Benchmarks & polish
1. `benches/ols_bench.rs` (criterion): olssm vs nalgebra QR vs normal equations,
   across `p ∈ {2, 5, 10, 20}` and `n ∈ {50, 500, 5000}`. Report honestly — the
   selling point is *no inversion / no square roots*, not necessarily raw speed.
   If it's competitive or faster in some regime, show it; if not, don't overclaim.
2. README (§5).
3. CHANGELOG.md (`0.1.0 — initial release`).

### Step D — CI
1. `.github/workflows/ci.yml`: matrix over OS (linux/mac/windows) × stable Rust;
   run `cargo test`, `clippy`, `fmt`; then `maturin develop` + `pytest`.
2. `.github/workflows/release.yml`: on tag, `maturin build --release` wheels for
   the platform matrix + `maturin publish` (PyPI token in secrets); separately
   `cargo publish` (crates.io token).

### Step E — Publish
1. `cargo publish --dry-run`, then `cargo publish` (name is permanent — confirm
   `olssm` is free on crates.io first; have a backup name).
2. `maturin publish` (or trigger the release workflow). Confirm `olssm` is free on
   PyPI.
3. Tag `v0.1.0`, write a short GitHub release note.

---

## 5. README contents (serves "citable" + "adoption")

- One-line description + the "no inversion, no normalization" hook.
- Install: `cargo add olssm` and `pip install olssm`.
- 5-line usage example in both Rust and Python.
- "What this implements" — bullet the paper's algorithms, link arXiv:2301.01854.
- **Citation block** (BibTeX for the Madar/Batista paper) + a note: *"If you use
  this implementation, please cite the original paper."*
- Benchmark summary table (honest).
- "When to use this vs `linfa-linear` / normal equations" — the differentiator is
  fidelity to the closed-forms and the square-root-free SGSO path, not a claim of
  beating LAPACK.
- License line: dual MIT/Apache-2.0.

---

## 6. Open questions to resolve during the build

1. **Re-derivation (§3) is the gating risk.** If the paper's βᵢ closed-form turns
   out to genuinely require the unit-diagonal step (i.e. the current code is
   right and the comment is just misleading), then the fix is only a *rename +
   doc correction*, not a logic change. Resolve by reading the paper's
   Algorithm/Theorem statements carefully and matching them line by line.
   Either way: the acceptance test (vs numpy) is the source of truth.
2. **f32 support?** The paper is f64-natural. Offer f32 only if there's demand;
   don't gold-plate v0.1.0.
3. **Intercept handling.** The paper gives forms "without assumption about
   modeling the intercept" plus an adjustment if needed. Decide whether `olssm`
   auto-augments a ones column or leaves it to the caller (recommend: leave to
   caller, document clearly, maybe a `solve_ols_with_intercept` helper).
4. **Crate name availability** on crates.io AND PyPI — check both before
   committing to `olssm`.

---

## 7. Quick checklist (for the build session)

- [ ] §3 correctness fix applied; acceptance test vs numpy passes
- [ ] Algs 1–3 + generalized_inverse ported, renamed, documented
- [ ] K-FAC/faer code excluded
- [ ] Rust tests + clippy + fmt green
- [ ] PyO3 bindings; `maturin develop` works; python tests pass
- [ ] Benchmarks run; README honest about speed
- [ ] Dual license files present; Cargo.toml + pyproject.toml license fields set
- [ ] CI green on linux/mac/windows
- [ ] Names free on crates.io + PyPI
- [ ] `cargo publish` + `maturin publish`; tag v0.1.0
```
