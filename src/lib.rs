//! `olsvered` — Closed-form OLS without inversion or normalisation.
//!
//! Exposes three algorithms from:
//! "Solving The Ordinary Least Squares in Closed Form, Without Inversion
//! or Normalization" — Vered Senderovich Madar & Sandra Batista.
//!
//! # Usage
//! - **Rust**: import `olsvered::algorithms::*` directly.
//! - **Python**: `import olsvered` after `maturin develop` / `pip install`.
//! - **C/C++**: link against `libolsvered` and include `olsvered.h`.

pub mod algorithms;
pub mod ffi;

// ---------------------------------------------------------------------------
// Python bindings (feature-gated — activated by maturin via extension-module)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use nalgebra::{DMatrix, DVector};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2, PyUntypedArrayMethods};
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
    // Module registration
    // -----------------------------------------------------------------------

    #[pymodule]
    pub fn olsvered(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(modified_cholesky, m)?)?;
        m.add_function(wrap_pyfunction!(back_substitute, m)?)?;
        m.add_function(wrap_pyfunction!(solve_ols, m)?)?;
        m.add_function(wrap_pyfunction!(simplified_gram_schmidt, m)?)?;
        m.add_function(wrap_pyfunction!(weighted_generalized_inverse, m)?)?;
        Ok(())
    }
}

#[cfg(feature = "python")]
pub use python_bindings::olsvered;
