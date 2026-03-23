# olsvered

**Closed-form Ordinary Least Squares — without matrix inversion or normalization.**

A Rust implementation of the three algorithms from the paper:

> **"Solving The Ordinary Least Squares in Closed Form, Without Inversion or Normalization"**
> Vered Senderovich Madar & Sandra L. Batista
> *arXiv:2301.01854* — [https://arxiv.org/abs/2301.01854](https://arxiv.org/abs/2301.01854)
> Submitted January 2023 · Revised December 2023

---

## Paper Highlights

Classical OLS solvers compute **β = (XᵀX)⁻¹Xᵀy**, which requires explicit matrix inversion — a numerically sensitive operation. This paper shows you can obtain the same coefficients **in closed form**, using only:

- **LU factorization** of the augmented Gram matrix `[X|y]ᵀ[X|y]`
- **Unnormalized Gram-Schmidt orthogonalization** (no square roots)

Key contributions:

| # | Algorithm | Core idea | Notable property |
|---|-----------|-----------|-----------------|
| 1 | **Modified Cholesky** | LU-decompose the Gram matrix, row-normalize the U factor | No explicit inversion; back-substitution recovers β |
| 2 | **Simplified Gram-Schmidt (SGSO)** | Orthogonalize columns of X without normalization | Zero square-root operations |
| 3 | **Weighted Generalized Inverse** | Compute `(XᵀWX)⁻¹XᵀW` via LU solve | Weighted OLS without inverting `XᵀWX` |

The paper establishes a direct algebraic connection between the LU upper-triangular factor and the Gram-Schmidt basis — giving an elegant unified view of two seemingly distinct decompositions.

---

## Installation

### Python (via maturin)

```bash
# In the repo root, with your virtual environment active
pip install maturin
maturin develop --features python
```

### Rust

```toml
# Cargo.toml
[dependencies]
olsvered = { path = "path/to/olsvered" }
```

### C / C++

Build the shared library and use the generated header:

```bash
cargo build --release
# Header auto-generated at: include/olsvered.h
```

---

## Python Demo

### Algorithm 1 — Modified Cholesky (standard OLS)

```python
import numpy as np
import olsvered

rng = np.random.default_rng(42)
X = rng.standard_normal((100, 4))          # 100 observations, 4 predictors
true_beta = np.array([1.0, -2.0, 3.0, 0.5])
y = X @ true_beta + rng.standard_normal(100) * 0.1   # small noise

# Solve OLS — no matrix inversion under the hood
beta = olsvered.solve_ols(X, y)
print("Estimated β:", beta)
print("True      β:", true_beta)
# → Estimated β: [ 1.002 -1.998  3.001  0.499]

# You can also inspect the intermediate C matrix (unit-diagonal upper triangular)
C = olsvered.modified_cholesky(X, y)
print("C shape:", C.shape)          # (5, 5) — augmented with y column
print("Diagonal:", np.diag(C))      # all 1.0
```

### Algorithm 2 — Simplified Gram-Schmidt (SGSO)

```python
import numpy as np
import olsvered

rng = np.random.default_rng(7)
X = rng.standard_normal((50, 5))

# Orthogonalize columns — no sqrt, no normalization
Q = olsvered.simplified_gram_schmidt(X)

print("Q shape:", Q.shape)           # (50, 5)

# Verify orthogonality: off-diagonal of QᵀQ should be ~0
QtQ = Q.T @ Q
off_diag = QtQ - np.diag(np.diag(QtQ))
print("Max off-diagonal:", np.abs(off_diag).max())   # ≈ 0.0
```

### Algorithm 3 — Weighted Generalized Inverse

```python
import numpy as np
import olsvered

rng = np.random.default_rng(99)
n, p = 30, 3
X = rng.standard_normal((n, p))
true_beta = np.array([2.0, -1.0, 0.5])
y = X @ true_beta + rng.standard_normal(n) * 0.05

# Diagonal weight matrix (e.g., inverse-variance weights)
W = np.diag(rng.uniform(0.5, 2.0, n))

# Compute weighted generalized inverse G = (XᵀWX)⁻¹ Xᵀ W
G = olsvered.weighted_generalized_inverse(X, W)
print("G shape:", G.shape)    # (3, 30) — shape (p, n)

# Weighted OLS solution
beta_w = G @ y
print("Weighted β:", beta_w)
print("True      β:", true_beta)
```

### Comparison with NumPy

```python
import numpy as np
import olsvered

rng = np.random.default_rng(0)
X = rng.standard_normal((200, 10))
y = rng.standard_normal(200)

beta_olsvered = olsvered.solve_ols(X, y)
beta_numpy, _, _, _ = np.linalg.lstsq(X, y, rcond=None)

print("Max abs difference:", np.abs(beta_olsvered - beta_numpy).max())
# → Max abs difference: ~1e-12
```

---

## Rust Demo

```rust
use nalgebra::{DMatrix, DVector};
use olsvered::algorithms::{solve_ols, simplified_gram_schmidt, weighted_generalized_inverse};

fn main() {
    // 5 observations, 2 predictors
    let x = DMatrix::from_row_slice(5, 2, &[
        1.0, 2.0,
        3.0, 4.0,
        5.0, 6.0,
        7.0, 8.0,
        9.0, 1.0,
    ]);
    let y = DVector::from_vec(vec![1.0, 2.0, 3.0, 4.0, 5.0]);

    // Algorithm 1 — standard OLS
    let beta = solve_ols(&x, &y).expect("OLS failed");
    println!("β = {:.4}", beta);

    // Algorithm 2 — SGSO orthogonalization
    let q = simplified_gram_schmidt(&x).expect("SGSO failed");
    println!("Q =\n{:.4}", q);

    // Algorithm 3 — weighted generalized inverse (W = I → standard pseudoinverse)
    let w = DMatrix::identity(5, 5);
    let g = weighted_generalized_inverse(&x, &w).expect("WGI failed");
    println!("G =\n{:.4}", g);
}
```

---

## Running Tests

```bash
# Rust unit tests (14 tests)
cargo test

# Python integration tests (21 tests) — requires maturin develop first
source /path/to/your/venv/bin/activate
maturin develop --features python
pytest tests/test_python.py -v
```

---

## Project Structure

```
olsvered/
├── src/
│   ├── algorithms.rs   # Pure Rust math — all 3 algorithms
│   ├── ffi.rs          # C ABI wrappers (extern "C")
│   └── lib.rs          # PyO3 Python bindings (feature-gated)
├── tests/
│   ├── test_algorithms.rs   # Rust unit tests
│   └── test_python.py       # Python integration tests vs numpy/scipy
├── include/
│   └── olsvered.h      # Auto-generated C header (cbindgen)
├── build.rs            # cbindgen build script
├── Cargo.toml
└── pyproject.toml      # maturin config
```

---

## Tech Stack

| Layer | Technology |
|-------|-----------|
| Core math | Rust · [nalgebra](https://nalgebra.org) 0.33 |
| Error handling | [thiserror](https://github.com/dtolnay/thiserror) 1.0 |
| Python bindings | [PyO3](https://pyo3.rs) 0.21 + [numpy](https://github.com/PyO3/rust-numpy) 0.21 |
| C/C++ bindings | [cbindgen](https://github.com/mozilla/cbindgen) 0.26 |
| Python build | [maturin](https://maturin.rs) |

---

## License

MIT
