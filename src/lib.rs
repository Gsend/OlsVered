//! `olssm` — Closed-form OLS without inversion or normalisation.
//!
//! Exposes three algorithms from:
//! or Normalization" —  Senderovich  & Sandra .
//!
//! # Usage
//! - **Rust**: import `olssm::algorithms::*` directly.
//! - **Python**: `import olssm` after `maturin develop` / `pip install`.
//! - **C/C++**: link against `libolssm` and include `olssm.h`.

pub mod algorithms;
pub mod ffi;

// ---------------------------------------------------------------------------
// Python bindings (feature-gated — activated by maturin via extension-module)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use nalgebra::{DMatrix, DVector};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods, Element};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use crate::algorithms;

    // -----------------------------------------------------------------------
    // Layout conversion helpers
    // -----------------------------------------------------------------------

    /// Convert a numpy (row-major / C-order) 2-D array to a nalgebra DMatrix.
    /// Always copies because nalgebra uses column-major storage internally.
    fn py_to_dmatrix(arr: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
        let shape = arr.shape();
        let slice = arr
            .as_slice()
            .expect("numpy array must be C-contiguous (call .copy() if needed)");
        DMatrix::from_row_slice(shape[0], shape[1], slice)
    }

    /// Convert a numpy 1-D array to a nalgebra DVector.
    fn py_to_dvector(arr: &PyReadonlyArray1<f64>) -> DVector<f64> {
        DVector::from_column_slice(
            arr.as_slice()
                .expect("numpy array must be contiguous"),
        )
    }

    // -----------------------------------------------------------------------
    // Python-exposed functions
    // -----------------------------------------------------------------------

    /// Algorithm 1: LU-based Gram matrix decomposition with row normalisation.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     y: numpy float64 array of shape (n,)
    ///
    /// Returns:
    ///     C matrix of shape (p+1, p+1), dtype float64, diagonal = 1.
    ///     Pass to ``back_substitute`` to recover OLS coefficients.
    ///
    /// Raises:
    ///     ValueError: on dimension mismatch or singular Gram matrix.
    #[pyfunction]
    pub fn modified_cholesky<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let yv = py_to_dvector(&y);
        let c = algorithms::modified_cholesky(&xm, &yv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = c.nrows();
        let cols = c.ncols();
        // Convert column-major nalgebra → row-major numpy
        let data: Vec<f64> = c.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Back-substitute C matrix to recover OLS coefficients.
    ///
    /// Args:
    ///     c: numpy float64 array of shape (p+1, p+1) — output of ``modified_cholesky``
    ///
    /// Returns:
    ///     beta: numpy float64 array of shape (p,)
    ///
    /// Raises:
    ///     ValueError: on invalid input.
    #[pyfunction]
    pub fn back_substitute<'py>(
        py: Python<'py>,
        c: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let cm = py_to_dmatrix(&c);
        let beta = algorithms::back_substitute(&cm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(beta.as_slice().to_vec().into_pyarray(py))
    }

    /// Full OLS solver — equivalent to ``modified_cholesky`` + ``back_substitute``.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     y: numpy float64 array of shape (n,)
    ///
    /// Returns:
    ///     beta: numpy float64 array of shape (p,), the OLS coefficients.
    ///
    /// Raises:
    ///     ValueError: on dimension mismatch or singular matrix.
    #[pyfunction]
    pub fn solve_ols<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        y: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let xm = py_to_dmatrix(&x);
        let yv = py_to_dvector(&y);
        let beta = algorithms::solve_ols(&xm, &yv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(beta.as_slice().to_vec().into_pyarray(py))
    }

    /// Algorithm 2: Non-normalised Gram-Schmidt orthogonalisation (SGSO).
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///
    /// Returns:
    ///     Q: numpy float64 array of shape (n, p) — orthogonal columns, not normalised.
    ///
    /// Raises:
    ///     ValueError: on invalid input.
    #[pyfunction]
    pub fn simplified_gram_schmidt<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let q = algorithms::simplified_gram_schmidt(&xm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = q.nrows();
        let cols = q.ncols();
        let data: Vec<f64> = q.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Algorithm 3: Weighted generalised inverse ``(XᵀWX)⁻¹ Xᵀ W``.
    ///
    /// Args:
    ///     x: numpy float64 array of shape (n, p)
    ///     w: numpy float64 array of shape (n, n), positive-definite weight matrix
    ///
    /// Returns:
    ///     G: numpy float64 array of shape (p, n).
    ///        Weighted OLS solution: ``beta = G @ y``.
    ///
    /// Raises:
    ///     ValueError: on dimension mismatch or singular matrix.
    #[pyfunction]
    pub fn weighted_generalized_inverse<'py>(
        py: Python<'py>,
        x: PyReadonlyArray2<'py, f64>,
        w: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let xm = py_to_dmatrix(&x);
        let wm = py_to_dmatrix(&w);
        let g = algorithms::weighted_generalized_inverse(&xm, &wm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = g.nrows();
        let cols = g.ncols();
        let data: Vec<f64> = g.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    // -----------------------------------------------------------------------
    // Algorithm 4 — Direct Gram matrix LU solve (K-FAC / Shampoo)
    // -----------------------------------------------------------------------

    /// Solve ``gram · X = rhs`` via LU factorisation — no explicit inverse.
    ///
    /// Core operation for second-order optimisers (K-FAC, Shampoo) that have
    /// a pre-computed Gram matrix and need to apply its inverse to a RHS.
    ///
    /// Args:
    ///     gram: numpy float64 array of shape (p, p), symmetric PSD
    ///     rhs:  numpy float64 array of shape (p, k)
    ///
    /// Returns:
    ///     X: numpy float64 array of shape (p, k) such that gram @ X ≈ rhs
    ///
    /// Raises:
    ///     ValueError: on dimension mismatch or singular matrix.
    #[pyfunction]
    pub fn lu_solve_gram<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let gm = py_to_dmatrix(&gram);
        let rm = py_to_dmatrix(&rhs);
        let result = algorithms::lu_solve_gram(&gm, &rm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = result.nrows();
        let cols = result.ncols();
        let data: Vec<f64> = result.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Solve ``gram · x = rhs`` for a single vector RHS.
    ///
    /// Args:
    ///     gram: numpy float64 array of shape (p, p)
    ///     rhs:  numpy float64 array of shape (p,)
    ///
    /// Returns:
    ///     x: numpy float64 array of shape (p,)
    #[pyfunction]
    pub fn lu_solve_gram_vec<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
        rhs: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let gm = py_to_dmatrix(&gram);
        let rv = py_to_dvector(&rhs);
        let result = algorithms::lu_solve_gram_vec(&gm, &rv)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(result.as_slice().to_vec().into_pyarray(py))
    }

    /// Compute ``gram⁻¹`` via LU factorisation.
    ///
    /// When K-FAC needs to cache the preconditioner for repeated application,
    /// an explicit inverse is appropriate. This computes it via LU rather than
    /// direct inversion for better numerical stability.
    ///
    /// Args:
    ///     gram: numpy float64 array of shape (p, p)
    ///
    /// Returns:
    ///     gram_inv: numpy float64 array of shape (p, p)
    #[pyfunction]
    pub fn lu_inverse_gram<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let gm = py_to_dmatrix(&gram);
        let result = algorithms::lu_inverse_gram(&gm)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        let rows = result.nrows();
        let cols = result.ncols();
        let data: Vec<f64> = result.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols])
            .map_err(|e| PyValueError::new_err(e.to_string()))
    }

    /// Fast K-FAC inverse: ``(gram + damping·I)⁻¹`` on f32 data.
    ///
    /// Compared with calling ``lu_inverse_gram`` on a float64 array, this
    /// function avoids the f32 → f64 dtype cast, the Tikhonov damping
    /// allocation, and the intermediate nalgebra matrix — reducing the
    /// per-call copy count from 6+ to 2.
    ///
    /// Args:
    ///     gram:    numpy float32 array of shape (n, n), C-contiguous
    ///     damping: scalar λ added to the diagonal before inversion
    ///
    /// Returns:
    ///     gram_inv: numpy float32 array of shape (n, n)
    #[pyfunction]
    pub fn lu_damped_inverse_f32<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f32>,
        damping: f64,
    ) -> PyResult<&'py PyArray2<f32>> {
        let shape = gram.shape();
        let n = shape[0];
        if shape[1] != n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                "gram must be square",
            ));
        }
        let slice = gram
            .as_slice()
            .expect("C-contiguous f32 array required; call .contiguous() first");
        let result = algorithms::lu_damped_inverse_f32(slice, n, damping as f32);
        let arr = PyArray1::from_vec(py, result);
        arr.reshape([n, n])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Compute the symmetric eigendecomposition of a Gram matrix (fast path for K-FAC).
    ///
    /// Uses faer's SIMD-accelerated self-adjoint EVD on f32 data.
    /// Damping is applied in eigenvalue space so Q can be reused when only
    /// changing the damping coefficient.
    ///
    /// Args:
    ///     gram   : numpy float32 array of shape (n, n), symmetric PSD
    ///     damping: scalar δ — returns 1 / max(λᵢ + δ, 1e-8)
    ///
    /// Returns:
    ///     (q, inv_lambda) where q is (n, n) float32 and inv_lambda is (n,) float32
    #[pyfunction]
    pub fn eigh_f32<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f32>,
        damping: f64,
    ) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
        let shape = gram.shape();
        let n = shape[0];
        if shape[1] != n {
            return Err(pyo3::exceptions::PyValueError::new_err("gram must be square"));
        }
        let slice = gram
            .as_slice()
            .expect("C-contiguous f32 array required");
        let (q_flat, inv_lam) = algorithms::eigh_f32(slice, n, damping as f32);
        let q_arr = PyArray1::from_vec(py, q_flat)
            .reshape([n, n])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let lam_arr = PyArray1::from_vec(py, inv_lam);
        Ok((q_arr, lam_arr))
    }

    /// Low-rank symmetric EVD: return top-k eigenvectors and damped inverse eigenvalues.
    ///
    /// Like ``eigh_f32`` but returns only the ``k`` eigenvectors for the largest eigenvalues,
    /// giving an n×k matrix Q_k.  The apply step then costs O((k_g+k_a)·d_out·d_in) instead
    /// of O(4·n·d_out·d_in), which is the genuine speedup over the full-rank eigen path.
    ///
    /// Args:
    ///     gram   : numpy float32 array of shape (n, n), symmetric PSD
    ///     k      : number of top eigenvectors to keep
    ///     damping: scalar δ — returns 1 / max(λᵢ + δ, 1e-8)
    ///
    /// Returns:
    ///     (q_k, inv_lambda_k) where q_k is (n, k) float32 and inv_lambda_k is (k,) float32
    #[pyfunction]
    pub fn eigh_topk_f32<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f32>,
        k: usize,
        damping: f64,
    ) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
        let shape = gram.shape();
        let n = shape[0];
        if shape[1] != n {
            return Err(pyo3::exceptions::PyValueError::new_err("gram must be square"));
        }
        if k == 0 || k > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("k must be in [1, n], got k={k}, n={n}"),
            ));
        }
        let slice = gram
            .as_slice()
            .expect("C-contiguous f32 array required");
        let (q_flat, inv_lam) = algorithms::eigh_topk_f32(slice, n, k, damping as f32);
        let q_arr = PyArray1::from_vec(py, q_flat)
            .reshape([n, k])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let lam_arr = PyArray1::from_vec(py, inv_lam);
        Ok((q_arr, lam_arr))
    }

    /// Apply the low-rank K-FAC preconditioner: ΔW ≈ Q_G_k d_G_k Q_G_kᵀ grad Q_A_k d_A_k Q_A_kᵀ
    ///
    /// Uses rank-k approximations of A and G, reducing apply cost from O(4n·d_out·d_in)
    /// to O((k_g+k_a)·d_out·d_in).  For k=32, n=512 this is ~16× fewer FLOPs.
    ///
    /// Args:
    ///     q_g_k      : (d_out, k_g) float32 top-k eigenvectors of G
    ///     inv_lam_g_k: (k_g,)       float32 damped inverse eigenvalues of G
    ///     grad       : (d_out, d_in) float32 weight gradient
    ///     q_a_k      : (d_in,  k_a) float32 top-k eigenvectors of A
    ///     inv_lam_a_k: (k_a,)       float32 damped inverse eigenvalues of A
    ///
    /// Returns:
    ///     (d_out, d_in) float32 preconditioned gradient
    #[pyfunction]
    pub fn apply_kfac_lowrank_f32<'py>(
        py: Python<'py>,
        q_g_k: PyReadonlyArray2<'py, f32>,
        inv_lam_g_k: PyReadonlyArray1<'py, f32>,
        grad: PyReadonlyArray2<'py, f32>,
        q_a_k: PyReadonlyArray2<'py, f32>,
        inv_lam_a_k: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let d_out = grad.shape()[0];
        let d_in  = grad.shape()[1];
        let k_g   = q_g_k.shape()[1];
        let k_a   = q_a_k.shape()[1];
        let qg_s  = q_g_k.as_slice().expect("C-contiguous q_g_k required");
        let qa_s  = q_a_k.as_slice().expect("C-contiguous q_a_k required");
        let gr_s  = grad.as_slice().expect("C-contiguous grad required");
        let lg_s  = inv_lam_g_k.as_slice().expect("C-contiguous inv_lam_g_k required");
        let la_s  = inv_lam_a_k.as_slice().expect("C-contiguous inv_lam_a_k required");
        let result = algorithms::apply_kfac_lowrank_f32(
            qg_s, k_g, lg_s,
            gr_s, d_out, d_in,
            qa_s, k_a, la_s,
        );
        PyArray1::from_vec(py, result)
            .reshape([d_out, d_in])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Apply the K-FAC eigen-basis preconditioner: ΔW = Q_G d_G Q_Gᵀ grad Q_A d_A Q_Aᵀ
    ///
    /// All 4 matrix products and the element-wise scaling are fused inside a single
    /// Rust call, eliminating Python dispatch overhead for the per-step hot path.
    ///
    /// Args:
    ///     q_g      : (d_out, d_out) float32 eigenvectors of G
    ///     inv_lam_g: (d_out,)       float32 damped inverse eigenvalues of G
    ///     grad     : (d_out, d_in)  float32 weight gradient
    ///     q_a      : (d_in,  d_in)  float32 eigenvectors of A
    ///     inv_lam_a: (d_in,)        float32 damped inverse eigenvalues of A
    ///
    /// Returns:
    ///     (d_out, d_in) float32 preconditioned gradient
    #[pyfunction]
    pub fn apply_kfac_eigen_f32<'py>(
        py: Python<'py>,
        q_g: PyReadonlyArray2<'py, f32>,
        inv_lam_g: PyReadonlyArray1<'py, f32>,
        grad: PyReadonlyArray2<'py, f32>,
        q_a: PyReadonlyArray2<'py, f32>,
        inv_lam_a: PyReadonlyArray1<'py, f32>,
    ) -> PyResult<&'py PyArray2<f32>> {
        let d_out = grad.shape()[0];
        let d_in  = grad.shape()[1];
        let qg_s = q_g.as_slice().expect("C-contiguous q_g required");
        let qa_s = q_a.as_slice().expect("C-contiguous q_a required");
        let gr_s = grad.as_slice().expect("C-contiguous grad required");
        let lg_s = inv_lam_g.as_slice().expect("C-contiguous inv_lam_g required");
        let la_s = inv_lam_a.as_slice().expect("C-contiguous inv_lam_a required");
        let result = algorithms::apply_kfac_eigen_f32(
            qg_s, lg_s, d_out,
            gr_s,
            qa_s, la_s, d_in,
        );
        PyArray1::from_vec(py, result)
            .reshape([d_out, d_in])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))
    }

    /// Randomized symmetric EVD — approximate top-k eigenvectors in O(k·n²).
    ///
    /// Uses the Halko-Martinsson-Tropp randomized range-finder:
    /// random projection → power iteration → Gram-Schmidt → small exact EVD.
    /// 24× cheaper than full EVD for k=32, n=784 with <1% error on K-FAC matrices.
    ///
    /// Args:
    ///     gram   : numpy float32 array of shape (n, n), symmetric PSD
    ///     k      : number of top eigenvectors to approximate
    ///     n_iter : power-iteration passes (default 1; use 2 for slowly-decaying spectra)
    ///     damping: scalar δ — returns 1 / max(λᵢ + δ, 1e-8)
    ///
    /// Returns:
    ///     (q_k, inv_lambda_k) where q_k is (n, k) float32 and inv_lambda_k is (k,) float32
    #[pyfunction]
    pub fn randomized_eigh_f32<'py>(
        py: Python<'py>,
        gram: PyReadonlyArray2<'py, f32>,
        k: usize,
        n_iter: usize,
        damping: f64,
    ) -> PyResult<(&'py PyArray2<f32>, &'py PyArray1<f32>)> {
        let shape = gram.shape();
        let n = shape[0];
        if shape[1] != n {
            return Err(pyo3::exceptions::PyValueError::new_err("gram must be square"));
        }
        if k == 0 || k > n {
            return Err(pyo3::exceptions::PyValueError::new_err(
                format!("k must be in [1, n], got k={k}, n={n}"),
            ));
        }
        let slice = gram.as_slice().expect("C-contiguous f32 array required");
        let (q_flat, inv_lam) =
            algorithms::randomized_eigh_f32(slice, n, k, n_iter, damping as f32);
        let q_arr = PyArray1::from_vec(py, q_flat)
            .reshape([n, k])
            .map_err(|e| pyo3::exceptions::PyValueError::new_err(e.to_string()))?;
        let lam_arr = PyArray1::from_vec(py, inv_lam);
        Ok((q_arr, lam_arr))
    }

    // -----------------------------------------------------------------------
    // Module registration
    // -----------------------------------------------------------------------

    #[pymodule]
    pub fn olssm(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(modified_cholesky, m)?)?;
        m.add_function(wrap_pyfunction!(back_substitute, m)?)?;
        m.add_function(wrap_pyfunction!(solve_ols, m)?)?;
        m.add_function(wrap_pyfunction!(simplified_gram_schmidt, m)?)?;
        m.add_function(wrap_pyfunction!(weighted_generalized_inverse, m)?)?;
        m.add_function(wrap_pyfunction!(lu_solve_gram, m)?)?;
        m.add_function(wrap_pyfunction!(lu_solve_gram_vec, m)?)?;
        m.add_function(wrap_pyfunction!(lu_inverse_gram, m)?)?;
        m.add_function(wrap_pyfunction!(lu_damped_inverse_f32, m)?)?;
        m.add_function(wrap_pyfunction!(eigh_f32, m)?)?;
        m.add_function(wrap_pyfunction!(apply_kfac_eigen_f32, m)?)?;
        m.add_function(wrap_pyfunction!(eigh_topk_f32, m)?)?;
        m.add_function(wrap_pyfunction!(apply_kfac_lowrank_f32, m)?)?;
        m.add_function(wrap_pyfunction!(randomized_eigh_f32, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::olssm;
