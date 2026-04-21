//! `olssm_vision` — SLAM / AR / SfM applications of the olssm closed-form
//! OLS kernels.
//!
//! This crate layers four capabilities on top of `olssm`:
//!
//! * `geometry` — SE(3) pose, pinhole projection, analytic Jacobians.
//! * `schur`    — BA reduced-camera system via Schur complement.
//! * `semi_partial` — whitened-Cholesky factor with semi-partial-correlation
//!   interpretation; gauge / degeneracy diagnostics.
//! * `selected_inverse` — on-demand columns / diagonal / blocks of `A⁻¹`
//!   for covariance recovery.
//!
//! # Usage (Rust)
//! ```no_run
//! use nalgebra::DMatrix;
//! use olssm_vision::{schur, semi_partial, selected_inverse};
//!
//! // (toy) SPD reduced camera Hessian
//! let s = DMatrix::<f64>::identity(6, 6) * 5.0;
//! let (lw, d) = semi_partial::whitened_cholesky(&s).unwrap();
//! let weak = semi_partial::detect_near_singular(&lw, 0.05);
//! assert!(weak.is_empty());
//! ```
//!
//! # Python bindings
//! Build with `maturin develop --features python` from the crate directory.

pub mod error;
pub mod geometry;
pub mod schur;
pub mod selected_inverse;
pub mod semi_partial;

pub use error::VisionError;

// ---------------------------------------------------------------------------
// Python bindings (feature-gated)
// ---------------------------------------------------------------------------

#[cfg(feature = "python")]
mod python_bindings {
    use nalgebra::{DMatrix, DVector};
    use numpy::{IntoPyArray, PyArray1, PyArray2, PyReadonlyArray1, PyReadonlyArray2};
    use pyo3::exceptions::PyValueError;
    use pyo3::prelude::*;

    use crate::{schur, selected_inverse, semi_partial};

    fn py_to_dmatrix(arr: &PyReadonlyArray2<f64>) -> DMatrix<f64> {
        let shape = arr.shape();
        let slice = arr.as_slice().expect("numpy array must be C-contiguous");
        DMatrix::from_row_slice(shape[0], shape[1], slice)
    }

    fn py_to_dvector(arr: &PyReadonlyArray1<f64>) -> DVector<f64> {
        DVector::from_column_slice(arr.as_slice().expect("numpy array must be contiguous"))
    }

    fn dmatrix_to_py<'py>(py: Python<'py>, m: &DMatrix<f64>) -> &'py PyArray2<f64> {
        let rows = m.nrows();
        let cols = m.ncols();
        let data: Vec<f64> = m.transpose().as_slice().to_vec();
        let arr = PyArray1::from_vec(py, data);
        arr.reshape([rows, cols]).unwrap()
    }

    fn dvector_to_py<'py>(py: Python<'py>, v: &DVector<f64>) -> &'py PyArray1<f64> {
        let data: Vec<f64> = v.iter().copied().collect();
        data.into_pyarray(py)
    }

    /// Build the Schur-reduced camera system. Returns a dict with
    /// `s`, `g_tilde`, `v_inv_block_diag`, and the cached `b`, `g_p`.
    #[pyfunction]
    #[pyo3(signature = (u, v, b, g_c, g_p, damping=0.0))]
    pub fn build_schur<'py>(
        py: Python<'py>,
        u: PyReadonlyArray2<'py, f64>,
        v: PyReadonlyArray2<'py, f64>,
        b: PyReadonlyArray2<'py, f64>,
        g_c: PyReadonlyArray1<'py, f64>,
        g_p: PyReadonlyArray1<'py, f64>,
        damping: f64,
    ) -> PyResult<&'py pyo3::types::PyDict> {
        let u = py_to_dmatrix(&u);
        let v = py_to_dmatrix(&v);
        let b = py_to_dmatrix(&b);
        let g_c = py_to_dvector(&g_c);
        let g_p = py_to_dvector(&g_p);

        let sys = schur::build_schur(&u, &v, &b, &g_c, &g_p, damping)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;

        // Assemble block-diagonal V_inv from per-point 3×3 blocks
        let n_pts = sys.v_inv_blocks.len();
        let mut v_inv = DMatrix::<f64>::zeros(3 * n_pts, 3 * n_pts);
        for (i, blk) in sys.v_inv_blocks.iter().enumerate() {
            let base = 3 * i;
            for r in 0..3 {
                for c in 0..3 {
                    v_inv[(base + r, base + c)] = blk[(r, c)];
                }
            }
        }

        let out = pyo3::types::PyDict::new(py);
        out.set_item("s", dmatrix_to_py(py, &sys.s))?;
        out.set_item("g_tilde", dvector_to_py(py, &sys.g_tilde))?;
        out.set_item("v_inv_block_diag", dmatrix_to_py(py, &v_inv))?;
        out.set_item("b", dmatrix_to_py(py, &sys.b))?;
        out.set_item("g_p", dvector_to_py(py, &sys.g_p))?;
        Ok(out)
    }

    /// Back-substitute point updates given camera update `delta_c`, the
    /// block-diagonal `V⁻¹`, original `B`, and `g_p`.
    #[pyfunction]
    pub fn backsubstitute_points<'py>(
        py: Python<'py>,
        delta_c: PyReadonlyArray1<'py, f64>,
        v_inv_block_diag: PyReadonlyArray2<'py, f64>,
        b: PyReadonlyArray2<'py, f64>,
        g_p: PyReadonlyArray1<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let delta_c = py_to_dvector(&delta_c);
        let v_inv = py_to_dmatrix(&v_inv_block_diag);
        let b = py_to_dmatrix(&b);
        let g_p = py_to_dvector(&g_p);

        let rhs = &g_p - b.transpose() * &delta_c;
        let dp = &v_inv * &rhs;
        Ok(dvector_to_py(py, &dp))
    }

    /// Whitened Cholesky factor of `a` plus its diagonal scales.
    /// Returns a tuple `(L_whitened, d)` where `L_whitened` is lower-triangular
    /// with semi-partial-correlation interpretation and `d[i] = sqrt(a[i, i])`.
    #[pyfunction]
    pub fn whitened_cholesky<'py>(
        py: Python<'py>,
        a: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<(&'py PyArray2<f64>, &'py PyArray1<f64>)> {
        let a = py_to_dmatrix(&a);
        let (lw, d) = semi_partial::whitened_cholesky(&a)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok((dmatrix_to_py(py, &lw), dvector_to_py(py, &d)))
    }

    /// Pivot profile: diagonal entries of the whitened Cholesky factor.
    /// Small values identify near-unobservable / gauge directions.
    #[pyfunction]
    pub fn pivot_profile<'py>(
        py: Python<'py>,
        l_whitened: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let lw = py_to_dmatrix(&l_whitened);
        let p = semi_partial::pivot_profile(&lw);
        Ok(dvector_to_py(py, &p))
    }

    /// List of `(column_index, pivot)` whose pivot is below `threshold`.
    #[pyfunction]
    pub fn detect_near_singular(
        l_whitened: PyReadonlyArray2<f64>,
        threshold: f64,
    ) -> PyResult<Vec<(usize, f64)>> {
        let lw = py_to_dmatrix(&l_whitened);
        Ok(semi_partial::detect_near_singular(&lw, threshold))
    }

    /// Sparsity template (list of `(i, j)` with `i ≥ j`) for an
    /// incomplete-Cholesky preconditioner guided by semi-partial-correlation
    /// magnitude.
    #[pyfunction]
    pub fn sparsity_template(
        l_whitened: PyReadonlyArray2<f64>,
        tolerance: f64,
    ) -> PyResult<Vec<(usize, usize)>> {
        let lw = py_to_dmatrix(&l_whitened);
        Ok(semi_partial::sparsity_template(&lw, tolerance))
    }

    /// `j`-th column of `A⁻¹` where `A = L Lᵀ`.
    #[pyfunction]
    pub fn covariance_column<'py>(
        py: Python<'py>,
        l: PyReadonlyArray2<'py, f64>,
        j: usize,
    ) -> PyResult<&'py PyArray1<f64>> {
        let l = py_to_dmatrix(&l);
        let col =
            selected_inverse::column(&l, j).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_py(py, &col))
    }

    /// Diagonal of `A⁻¹` via Takahashi recursion.
    #[pyfunction]
    pub fn covariance_diagonal<'py>(
        py: Python<'py>,
        l: PyReadonlyArray2<'py, f64>,
    ) -> PyResult<&'py PyArray1<f64>> {
        let l = py_to_dmatrix(&l);
        let d =
            selected_inverse::diagonal(&l).map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dvector_to_py(py, &d))
    }

    /// Symmetric `k × k` block of `A⁻¹` at the given indices.
    #[pyfunction]
    pub fn covariance_block<'py>(
        py: Python<'py>,
        l: PyReadonlyArray2<'py, f64>,
        indices: Vec<usize>,
    ) -> PyResult<&'py PyArray2<f64>> {
        let l = py_to_dmatrix(&l);
        let blk = selected_inverse::block(&l, &indices)
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        Ok(dmatrix_to_py(py, &blk))
    }

    #[pymodule]
    fn _rust(_py: Python, m: &PyModule) -> PyResult<()> {
        m.add_function(wrap_pyfunction!(build_schur, m)?)?;
        m.add_function(wrap_pyfunction!(backsubstitute_points, m)?)?;
        m.add_function(wrap_pyfunction!(whitened_cholesky, m)?)?;
        m.add_function(wrap_pyfunction!(pivot_profile, m)?)?;
        m.add_function(wrap_pyfunction!(detect_near_singular, m)?)?;
        m.add_function(wrap_pyfunction!(sparsity_template, m)?)?;
        m.add_function(wrap_pyfunction!(covariance_column, m)?)?;
        m.add_function(wrap_pyfunction!(covariance_diagonal, m)?)?;
        m.add_function(wrap_pyfunction!(covariance_block, m)?)?;
        Ok(())
    }
}
