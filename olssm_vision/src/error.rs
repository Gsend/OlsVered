//! Error types shared across `olssm_vision`.

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum VisionError {
    #[error("Dimension mismatch: expected {expected}, got {got} ({what})")]
    DimensionMismatch {
        expected: usize,
        got: usize,
        what: &'static str,
    },

    #[error("Point block V[{index}] is singular (det ≈ 0); cannot Schur-eliminate")]
    SingularPointBlock { index: usize },

    #[error("Reduced camera system is rank-deficient; add damping or check gauge")]
    RankDeficientS,

    #[error("Pivot at column {index} below threshold {threshold:.3e} — near-singular direction")]
    NearSingularPivot { index: usize, threshold: f64 },

    #[error("Underlying olssm core error: {0}")]
    Core(String),
}

impl From<olssm::algorithms::OlsSMError> for VisionError {
    fn from(e: olssm::algorithms::OlsSMError) -> Self {
        VisionError::Core(e.to_string())
    }
}
