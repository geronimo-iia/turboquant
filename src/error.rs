use std::fmt;

/// Error type for all qjl-sketch operations.
#[derive(Debug)]
pub enum QjlError {
    /// Input vector length does not match expected dimension.
    DimensionMismatch { expected: usize, got: usize },
    /// Sketch dimension is invalid (must be > 0 and divisible by 8).
    InvalidSketchDim(usize),
    /// Bit width is invalid (must be 2 or 4).
    InvalidBitWidth(u8),
    /// Input contains non-finite values (NaN or infinity).
    NonFiniteInput { context: &'static str },
    /// Store file has wrong magic bytes.
    StoreMagicMismatch,
    /// Store file has unsupported version.
    StoreVersionMismatch { expected: u16, got: u16 },
    /// Outlier index out of range.
    OutlierIndexOutOfRange { index: u8, head_dim: usize },
    /// I/O error from the underlying filesystem.
    Io(std::io::Error),
}

/// Convenience alias.
pub type Result<T> = std::result::Result<T, QjlError>;

impl fmt::Display for QjlError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::DimensionMismatch { expected, got } => {
                write!(f, "dimension mismatch: expected {expected}, got {got}")
            }
            Self::InvalidSketchDim(dim) => {
                write!(
                    f,
                    "invalid sketch dimension: {dim} (must be > 0 and divisible by 8)"
                )
            }
            Self::InvalidBitWidth(bits) => {
                write!(f, "invalid bit width: {bits} (must be 2 or 4)")
            }
            Self::NonFiniteInput { context } => {
                write!(f, "non-finite input in {context}")
            }
            Self::StoreMagicMismatch => {
                write!(f, "store file has wrong magic bytes")
            }
            Self::StoreVersionMismatch { expected, got } => {
                write!(f, "store version mismatch: expected {expected}, got {got}")
            }
            Self::OutlierIndexOutOfRange { index, head_dim } => {
                write!(
                    f,
                    "outlier index {index} out of range for head_dim {head_dim}"
                )
            }
            Self::Io(e) => write!(f, "I/O error: {e}"),
        }
    }
}

impl std::error::Error for QjlError {
    fn source(&self) -> Option<&(dyn std::error::Error + 'static)> {
        match self {
            Self::Io(e) => Some(e),
            _ => None,
        }
    }
}

impl From<std::io::Error> for QjlError {
    fn from(e: std::io::Error) -> Self {
        Self::Io(e)
    }
}

/// Check that all values in a slice are finite.
pub fn validate_finite(values: &[f32], context: &'static str) -> Result<()> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(QjlError::NonFiniteInput { context });
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_display_dimension_mismatch() {
        let e = QjlError::DimensionMismatch {
            expected: 128,
            got: 64,
        };
        assert_eq!(e.to_string(), "dimension mismatch: expected 128, got 64");
    }

    #[test]
    fn test_display_all_variants() {
        // Just ensure Display doesn't panic on any variant
        let variants: Vec<QjlError> = vec![
            QjlError::DimensionMismatch {
                expected: 1,
                got: 2,
            },
            QjlError::InvalidSketchDim(7),
            QjlError::InvalidBitWidth(3),
            QjlError::NonFiniteInput { context: "test" },
            QjlError::StoreMagicMismatch,
            QjlError::StoreVersionMismatch {
                expected: 1,
                got: 2,
            },
            QjlError::OutlierIndexOutOfRange {
                index: 200,
                head_dim: 128,
            },
            QjlError::Io(std::io::Error::new(std::io::ErrorKind::NotFound, "gone")),
        ];
        for v in &variants {
            assert!(!v.to_string().is_empty());
        }
    }

    #[test]
    fn test_from_io_error() {
        let io_err = std::io::Error::new(std::io::ErrorKind::NotFound, "not found");
        let qjl_err: QjlError = io_err.into();
        assert!(matches!(qjl_err, QjlError::Io(_)));
    }

    #[test]
    fn test_validate_finite_ok() {
        assert!(validate_finite(&[1.0, 2.0, 3.0], "test").is_ok());
    }

    #[test]
    fn test_validate_finite_nan() {
        assert!(validate_finite(&[1.0, f32::NAN, 3.0], "test").is_err());
    }

    #[test]
    fn test_validate_finite_inf() {
        assert!(validate_finite(&[f32::INFINITY], "test").is_err());
    }

    #[test]
    fn test_error_source() {
        let io_err = std::io::Error::new(std::io::ErrorKind::Other, "oops");
        let qjl_err = QjlError::Io(io_err);
        assert!(std::error::Error::source(&qjl_err).is_some());

        let dim_err = QjlError::DimensionMismatch {
            expected: 1,
            got: 2,
        };
        assert!(std::error::Error::source(&dim_err).is_none());
    }
}
