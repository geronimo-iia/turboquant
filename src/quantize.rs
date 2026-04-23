use crate::error::{validate_finite, QjlError, Result};
use crate::outliers::outlier_mask;
use crate::sketch::{l2_norm, QJLSketch};

/// Compressed key vectors — packed sign bits with outlier separation.
#[derive(Clone, Debug)]
pub struct CompressedKeys {
    /// Packed sign bits for inlier dimensions [num_vectors, sketch_dim/8].
    pub key_quant: Vec<u8>,
    /// Packed sign bits for outlier dimensions [num_vectors, outlier_sketch_dim/8].
    pub key_outlier_quant: Vec<u8>,
    /// L2 norm of each full vector [num_vectors].
    pub key_norms: Vec<f32>,
    /// L2 norm of outlier components [num_vectors].
    pub outlier_norms: Vec<f32>,
    /// Outlier dimension indices for this group [outlier_count].
    pub outlier_indices: Vec<u8>,
    /// Number of vectors.
    pub num_vectors: usize,
    /// Head dimension.
    pub head_dim: usize,
}

impl QJLSketch {
    /// Quantize a group of key vectors into packed sign bits.
    ///
    /// - `keys`: flattened [num_vectors, head_dim] row-major
    /// - `num_vectors`: number of vectors
    /// - `outlier_indices`: which dimensions are outliers for this group
    pub fn quantize(
        &self,
        keys: &[f32],
        num_vectors: usize,
        outlier_indices: &[u8],
    ) -> Result<CompressedKeys> {
        let d = self.head_dim;
        let s = self.sketch_dim;
        let os = self.outlier_sketch_dim;
        if keys.len() != num_vectors * d {
            return Err(QjlError::DimensionMismatch {
                expected: num_vectors * d,
                got: keys.len(),
            });
        }
        validate_finite(keys, "quantize keys")?;
        for &idx in outlier_indices {
            if idx as usize >= d {
                return Err(QjlError::OutlierIndexOutOfRange {
                    index: idx,
                    head_dim: d,
                });
            }
        }

        let mask = outlier_mask(outlier_indices, d);
        let inlier_bytes = s / 8;
        let outlier_bytes = os / 8;

        let mut key_quant = vec![0u8; num_vectors * inlier_bytes];
        let mut key_outlier_quant = vec![0u8; num_vectors * outlier_bytes];
        let mut key_norms = vec![0.0f32; num_vectors];
        let mut outlier_norms = vec![0.0f32; num_vectors];

        for v in 0..num_vectors {
            let vec_start = v * d;
            let vec_data = &keys[vec_start..vec_start + d];

            // Full vector norm
            key_norms[v] = l2_norm(vec_data);

            // Outlier norm
            let mut outlier_sq = 0.0f32;
            for &idx in outlier_indices {
                let val = vec_data[idx as usize];
                outlier_sq += val * val;
            }
            outlier_norms[v] = outlier_sq.sqrt();

            // Project and extract signs for inliers and outliers
            // proj_dir_quant is [s, d], each row is a projection direction
            let quant_offset = v * inlier_bytes;
            let outlier_quant_offset = v * outlier_bytes;

            for p in 0..s {
                let proj_row = &self.proj_dir_quant[p * d..(p + 1) * d];

                // Inlier projection
                let mut dot_inlier = 0.0f32;
                let mut dot_outlier = 0.0f32;
                for i in 0..d {
                    let contrib = proj_row[i] * vec_data[i];
                    if mask[i] {
                        dot_outlier += contrib;
                    } else {
                        dot_inlier += contrib;
                    }
                }

                // Pack inlier sign bit
                let byte_idx = p / 8;
                let bit_idx = p % 8;
                if dot_inlier > 0.0 {
                    key_quant[quant_offset + byte_idx] |= 1 << bit_idx;
                }

                // Pack outlier sign bit (only for first os projections)
                if p < os {
                    let ob_idx = p / 8;
                    let ob_bit = p % 8;
                    if dot_outlier > 0.0 {
                        key_outlier_quant[outlier_quant_offset + ob_idx] |= 1 << ob_bit;
                    }
                }
            }
        }

        Ok(CompressedKeys {
            key_quant,
            key_outlier_quant,
            key_norms,
            outlier_norms,
            outlier_indices: outlier_indices.to_vec(),
            num_vectors,
            head_dim: d,
        })
    }
}

/// Pack sign bits: for each group of 8 booleans, produce one u8.
pub fn pack_signs(signs: &[bool]) -> Vec<u8> {
    let num_bytes = signs.len().div_ceil(8);
    let mut packed = vec![0u8; num_bytes];
    for (i, &sign) in signs.iter().enumerate() {
        if sign {
            packed[i / 8] |= 1 << (i % 8);
        }
    }
    packed
}

/// Unpack sign bits: expand packed bytes back to booleans.
pub fn unpack_signs(packed: &[u8], count: usize) -> Vec<bool> {
    let mut signs = Vec::with_capacity(count);
    for i in 0..count {
        signs.push((packed[i / 8] >> (i % 8)) & 1 == 1);
    }
    signs
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_pack_unpack_roundtrip() {
        let signs = vec![true, false, true, true, false, false, true, false, true];
        let packed = pack_signs(&signs);
        let unpacked = unpack_signs(&packed, signs.len());
        assert_eq!(signs, unpacked);
    }

    #[test]
    fn test_pack_known_byte() {
        // bits: 1,0,1,1,0,0,1,0 → 0b01001101 = 0x4D = 77
        let signs = vec![true, false, true, true, false, false, true, false];
        let packed = pack_signs(&signs);
        assert_eq!(packed, vec![0x4D]);
    }

    #[test]
    fn test_quantize_output_shape() {
        let sketch = QJLSketch::new(16, 32, 16, 42).unwrap();
        let num_vectors = 4;
        let keys = vec![1.0f32; num_vectors * 16];
        let outlier_indices = vec![0u8, 1];

        let compressed = sketch
            .quantize(&keys, num_vectors, &outlier_indices)
            .unwrap();

        assert_eq!(compressed.key_quant.len(), num_vectors * (32 / 8));
        assert_eq!(compressed.key_outlier_quant.len(), num_vectors * (16 / 8));
        assert_eq!(compressed.key_norms.len(), num_vectors);
        assert_eq!(compressed.outlier_norms.len(), num_vectors);
        assert_eq!(compressed.num_vectors, num_vectors);
    }

    #[test]
    fn test_quantize_norms() {
        let sketch = QJLSketch::new(4, 8, 8, 42).unwrap();
        let keys = vec![3.0, 0.0, 4.0, 0.0]; // norm = 5
        let outlier_indices = vec![2u8]; // dim 2 is outlier (value=4)

        let compressed = sketch.quantize(&keys, 1, &outlier_indices).unwrap();

        assert!((compressed.key_norms[0] - 5.0).abs() < 1e-6);
        assert!((compressed.outlier_norms[0] - 4.0).abs() < 1e-6);
    }

    #[test]
    fn test_quantize_outlier_separation() {
        let d = 8;
        let s = 16;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();

        // Vector with all energy in dim 0 (outlier) and dim 4 (inlier)
        let mut keys = vec![0.0f32; d];
        keys[0] = 10.0; // outlier
        keys[4] = 5.0; // inlier

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, 1, &outlier_indices).unwrap();

        // Outlier norm should be 10.0
        assert!((compressed.outlier_norms[0] - 10.0).abs() < 1e-6);
        // Full norm should be sqrt(100+25) = sqrt(125)
        assert!((compressed.key_norms[0] - 125.0f32.sqrt()).abs() < 1e-5);
    }
}
