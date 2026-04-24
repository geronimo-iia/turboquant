use nalgebra::DMatrix;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

use crate::error::{QjlError, Result};

/// Random projection sketch for QJL quantization.
///
/// Generates an orthogonalized random projection matrix used for
/// sign-based vector compression. The projection preserves inner
/// product structure (Johnson-Lindenstrauss).
pub struct QJLSketch {
    pub head_dim: usize,
    pub sketch_dim: usize,
    pub outlier_sketch_dim: usize,
    /// Orthogonalized projection matrix [head_dim, sketch_dim], row-major.
    /// Used to sketch queries: sketched_q = query @ proj_dir_score.
    pub proj_dir_score: Vec<f32>,
    /// Transposed projection [sketch_dim, head_dim], row-major.
    /// Used to sketch keys: sketched_k = proj_dir_quant @ key.
    pub proj_dir_quant: Vec<f32>,
}

impl QJLSketch {
    /// Create a new sketch with the given dimensions and RNG seed.
    ///
    /// - `head_dim`: dimension of each vector (d)
    /// - `sketch_dim`: number of random projections (s), must be divisible by 8
    /// - `outlier_sketch_dim`: sketch dimension for outlier components, must be divisible by 8
    /// - `seed`: RNG seed for reproducibility
    pub fn new(
        head_dim: usize,
        sketch_dim: usize,
        outlier_sketch_dim: usize,
        seed: u64,
    ) -> Result<Self> {
        if head_dim == 0 {
            return Err(QjlError::DimensionMismatch {
                expected: 1,
                got: 0,
            });
        }
        if sketch_dim == 0 || !sketch_dim.is_multiple_of(8) {
            return Err(QjlError::InvalidSketchDim(sketch_dim));
        }
        if !outlier_sketch_dim.is_multiple_of(8) {
            return Err(QjlError::InvalidSketchDim(outlier_sketch_dim));
        }
        if outlier_sketch_dim > sketch_dim {
            return Err(QjlError::InvalidSketchDim(outlier_sketch_dim));
        }

        let mut rng = ChaCha20Rng::seed_from_u64(seed);
        let proj_dir_score = init_orthogonal_projection(head_dim, sketch_dim, &mut rng);

        // proj_dir_quant is the transpose: [sketch_dim, head_dim]
        let proj_dir_quant = transpose(&proj_dir_score, head_dim, sketch_dim);

        Ok(Self {
            head_dim,
            sketch_dim,
            outlier_sketch_dim,
            proj_dir_score,
            proj_dir_quant,
        })
    }
}

/// Generate an orthogonalized random projection matrix [head_dim, sketch_dim].
///
/// Algorithm:
/// 1. Sample Gaussian random matrix [head_dim, sketch_dim]
/// 2. Split into chunks of [head_dim, head_dim] (or smaller for the last chunk)
/// 3. QR-decompose each chunk
/// 4. Replace chunk with Q * sqrt(head_dim)
fn init_orthogonal_projection(d: usize, s: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    // Sample random Gaussian matrix [d, s] stored as column-major for nalgebra
    let normal = StandardNormal;
    let data: Vec<f64> = (0..d * s).map(|_| normal.sample(rng)).collect();
    // nalgebra DMatrix is column-major: from_vec fills column by column
    let mut mat = DMatrix::from_vec(d, s, data);

    // QR orthogonalize in chunks of d columns
    let num_chunks = s.div_ceil(d);
    let scale = (d as f64).sqrt();

    for i in 0..num_chunks {
        let col_start = i * d;
        let col_end = (col_start + d).min(s);
        let chunk_cols = col_end - col_start;

        let chunk = mat.columns(col_start, chunk_cols).clone_owned();
        let qr = chunk.qr();
        let q = qr.q();

        for c in 0..chunk_cols {
            for r in 0..d {
                mat[(r, col_start + c)] = q[(r, c)] * scale;
            }
        }
    }

    // Convert to row-major [d, s] for our storage
    let mut result = vec![0.0f32; d * s];
    for r in 0..d {
        for c in 0..s {
            result[r * s + c] = mat[(r, c)] as f32;
        }
    }
    result
}

/// Transpose a row-major matrix [rows, cols] → [cols, rows].
fn transpose(data: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut out = vec![0.0f32; rows * cols];
    for r in 0..rows {
        for c in 0..cols {
            out[c * rows + r] = data[r * cols + c];
        }
    }
    out
}

/// Compute the matrix-vector product: result = mat @ vec.
/// mat is \[rows, cols\] row-major, vec is \[cols\].
#[allow(clippy::needless_range_loop)]
pub fn matvec(mat: &[f32], rows: usize, cols: usize, vec: &[f32]) -> Vec<f32> {
    assert_eq!(mat.len(), rows * cols);
    assert_eq!(vec.len(), cols);
    let mut out = vec![0.0f32; rows];
    for r in 0..rows {
        let row_start = r * cols;
        let mut acc = 0.0f32;
        for c in 0..cols {
            acc += mat[row_start + c] * vec[c];
        }
        out[r] = acc;
    }
    out
}

/// Compute the vector L2 norm.
pub fn l2_norm(v: &[f32]) -> f32 {
    v.iter().map(|x| x * x).sum::<f32>().sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proj_dimensions() {
        let sketch = QJLSketch::new(128, 256, 64, 42).unwrap();
        assert_eq!(sketch.proj_dir_score.len(), 128 * 256);
        assert_eq!(sketch.proj_dir_quant.len(), 256 * 128);
        assert_eq!(sketch.head_dim, 128);
        assert_eq!(sketch.sketch_dim, 256);
        assert_eq!(sketch.outlier_sketch_dim, 64);
    }

    #[test]
    fn test_proj_orthogonality() {
        let d = 64;
        let s = 128;
        let sketch = QJLSketch::new(d, s, 32, 42).unwrap();

        // First d×d chunk should be approximately orthogonal:
        // Q^T Q ≈ d * I  (because we scaled by sqrt(d))
        let chunk: DMatrix<f64> =
            DMatrix::from_fn(d, d, |r, c| sketch.proj_dir_score[r * s + c] as f64);
        let product = chunk.transpose() * &chunk;

        for r in 0..d {
            for c in 0..d {
                if r == c {
                    assert!(
                        (product[(r, c)] - d as f64).abs() < 0.1,
                        "diagonal [{r},{c}]: expected {d}, got {}",
                        product[(r, c)]
                    );
                } else {
                    assert!(
                        product[(r, c)].abs() < 0.1,
                        "off-diagonal [{r},{c}]: expected 0, got {}",
                        product[(r, c)]
                    );
                }
            }
        }
    }

    #[test]
    fn test_proj_deterministic() {
        let a = QJLSketch::new(64, 128, 32, 42).unwrap();
        let b = QJLSketch::new(64, 128, 32, 42).unwrap();
        assert_eq!(a.proj_dir_score, b.proj_dir_score);
        assert_eq!(a.proj_dir_quant, b.proj_dir_quant);
    }

    #[test]
    fn test_proj_different_seeds() {
        let a = QJLSketch::new(64, 128, 32, 42).unwrap();
        let b = QJLSketch::new(64, 128, 32, 99).unwrap();
        assert_ne!(a.proj_dir_score, b.proj_dir_score);
    }

    #[test]
    fn test_new_zero_head_dim() {
        assert!(QJLSketch::new(0, 128, 32, 42).is_err());
    }

    #[test]
    fn test_new_sketch_dim_not_divisible_by_8() {
        assert!(QJLSketch::new(64, 100, 32, 42).is_err());
    }

    #[test]
    fn test_new_outlier_dim_exceeds_sketch_dim() {
        assert!(QJLSketch::new(64, 128, 256, 42).is_err());
    }

    #[test]
    fn test_transpose_roundtrip() {
        let d = 4;
        let s = 8;
        let data: Vec<f32> = (0..d * s).map(|i| i as f32).collect();
        let t = transpose(&data, d, s);
        let tt = transpose(&t, s, d);
        assert_eq!(data, tt);
    }

    #[test]
    fn test_matvec() {
        // 2x3 matrix times 3-vector
        let mat = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0];
        let v = vec![1.0, 1.0, 1.0];
        let result = matvec(&mat, 2, 3, &v);
        assert_eq!(result, vec![6.0, 15.0]);
    }
}
