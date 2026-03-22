#include <cstdarg>
#include <cstdint>
#include <cstdlib>
#include <ostream>
#include <new>

/// Status codes returned by all `olsvered_*` FFI functions.
enum class OlsveredStatus {
  /// Success.
  Ok = 0,
  /// Row/column dimension mismatch between inputs.
  DimensionMismatch = 1,
  /// Zero pivot encountered — input is (near-)singular.
  ZeroPivot = 2,
  /// A required pointer argument was NULL.
  NullPointer = 3,
  /// Matrix is singular; solve returned no solution.
  SingularMatrix = 4,
};

extern "C" {

/// Algorithm 1: Modified Cholesky.
///
/// @param x        Column-major f64 array, shape (x_rows × x_cols)
/// @param x_rows   Number of rows in X (n samples)
/// @param x_cols   Number of columns in X (p predictors)
/// @param y        f64 array of length x_rows
/// @param c_out    Caller-allocated output buffer of size (x_cols+1)*(x_cols+1)
/// @return         OlsveredStatus::Ok on success
OlsveredStatus olsvered_modified_cholesky(const double *x,
                                          uintptr_t x_rows,
                                          uintptr_t x_cols,
                                          const double *y,
                                          double *c_out);

/// Back-substitute C to recover OLS beta coefficients.
///
/// @param c        Column-major f64 array, shape (dim × dim)
/// @param dim      Dimension of C (= p+1)
/// @param beta_out Caller-allocated output buffer of length (dim-1)
/// @return         OlsveredStatus::Ok on success
OlsveredStatus olsvered_back_substitute(const double *c, uintptr_t dim, double *beta_out);

/// Full OLS solver (Algorithm 1 + back-substitution).
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param y        f64 array of length x_rows
/// @param beta_out Caller-allocated output buffer of length x_cols
/// @return         OlsveredStatus::Ok on success
OlsveredStatus olsvered_solve_ols(const double *x,
                                  uintptr_t x_rows,
                                  uintptr_t x_cols,
                                  const double *y,
                                  double *beta_out);

/// Algorithm 2: Simplified (non-normalised) Gram-Schmidt orthogonalisation.
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param q_out    Caller-allocated output buffer (x_rows * x_cols)
/// @return         OlsveredStatus::Ok on success
OlsveredStatus olsvered_simplified_gram_schmidt(const double *x,
                                                uintptr_t x_rows,
                                                uintptr_t x_cols,
                                                double *q_out);

/// Algorithm 3: Weighted generalised inverse (XᵀWX)⁻¹ Xᵀ W.
///
/// @param x        Column-major f64 array (x_rows × x_cols)
/// @param x_rows   n samples
/// @param x_cols   p predictors
/// @param w        Column-major f64 array (x_rows × x_rows) — weight matrix
/// @param g_out    Caller-allocated output buffer (x_cols * x_rows)
/// @return         OlsveredStatus::Ok on success
OlsveredStatus olsvered_weighted_generalized_inverse(const double *x,
                                                     uintptr_t x_rows,
                                                     uintptr_t x_cols,
                                                     const double *w,
                                                     double *g_out);

} // extern "C"
