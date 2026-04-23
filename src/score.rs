use crate::error::{validate_finite, QjlError, Result};
use crate::quantize::CompressedKeys;
use crate::sketch::{matvec, QJLSketch};

impl QJLSketch {
    /// Compute approximate attention scores between a query and compressed keys.
    ///
    /// Matches the QJL CUDA kernel (`calc_score_kernel`):
    ///   q_sketch = query @ proj_dir_score  (full sketch)
    ///   q_outlier_sketch[i] = Σ_j query[outlier_j] * proj_dir_score[outlier_j, i]
    ///   q_inlier_sketch = q_sketch - q_outlier_sketch
    ///   score = sqrt(π/2)/s * ||k_inlier|| * Σ sign(k_inlier_i) * q_inlier_sketch[i]
    ///         + sqrt(π/2)/os * ||k_outlier|| * Σ sign(k_outlier_i) * q_outlier_sketch[i]
    ///
    /// - `query`: [head_dim] f32
    /// - `compressed`: compressed key vectors from `quantize()`
    pub fn score(&self, query: &[f32], compressed: &CompressedKeys) -> Result<Vec<f32>> {
        let d = self.head_dim;
        let s = self.sketch_dim;
        let os = self.outlier_sketch_dim;
        if query.len() != d {
            return Err(QjlError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }
        validate_finite(query, "score query")?;

        let inlier_bytes = s / 8;
        let outlier_bytes = os / 8;

        // Full query sketch: q_sketch = proj_dir_quant @ query → [s]
        let q_sketch = matvec(&self.proj_dir_quant, s, d, query);

        // Outlier query sketch: only the outlier dimensions of query projected
        // q_outlier_sketch[p] = Σ_j query[outlier_j] * proj_dir_score[outlier_j, p]
        // proj_dir_score is [d, s] row-major, so proj_dir_score[j, p] = proj_dir_score[j*s + p]
        let mut q_outlier_sketch = vec![0.0f32; s];
        for &idx in &compressed.outlier_indices {
            let j = idx as usize;
            let row_start = j * s;
            for (p, qos) in q_outlier_sketch.iter_mut().enumerate().take(s) {
                *qos += query[j] * self.proj_dir_score[row_start + p];
            }
        }

        // Inlier query sketch = full - outlier
        let q_inlier_sketch: Vec<f32> = q_sketch
            .iter()
            .zip(q_outlier_sketch.iter())
            .map(|(full, outlier)| full - outlier)
            .collect();

        // Scale factors from the CUDA kernel: sqrt(π/2) / sketch_dim
        let scl = (std::f32::consts::FRAC_PI_2).sqrt() / s as f32;
        let scl_outlier = (std::f32::consts::FRAC_PI_2).sqrt() / os as f32;

        let mut scores = vec![0.0f32; compressed.num_vectors];

        for (v, score) in scores.iter_mut().enumerate().take(compressed.num_vectors) {
            // Inlier: signed dot of q_inlier_sketch against key inlier sign bits
            let k_inlier = &compressed.key_quant[v * inlier_bytes..(v + 1) * inlier_bytes];
            let dot_inlier = signed_dot(&q_inlier_sketch, k_inlier, s);

            let full_sq = compressed.key_norms[v] * compressed.key_norms[v];
            let outlier_sq = compressed.outlier_norms[v] * compressed.outlier_norms[v];
            let inlier_norm = (full_sq - outlier_sq).max(0.0).sqrt();

            // Outlier: signed dot of q_outlier_sketch against key outlier sign bits
            let k_outlier =
                &compressed.key_outlier_quant[v * outlier_bytes..(v + 1) * outlier_bytes];
            let dot_outlier = signed_dot(&q_outlier_sketch[..os], k_outlier, os);

            *score = scl * inlier_norm * dot_inlier
                + scl_outlier * compressed.outlier_norms[v] * dot_outlier;
        }

        Ok(scores)
    }
}

/// Compute the dot product between float query sketch and packed sign bits.
///
/// For each projection i: result += sketched_q[i] * sign(sketched_k[i])
/// where sign is +1 if bit is set, -1 if not.
#[allow(clippy::needless_range_loop)]
fn signed_dot(sketched_q: &[f32], packed_signs: &[u8], total_bits: usize) -> f32 {
    let mut acc = 0.0f32;
    for i in 0..total_bits {
        let byte_idx = i / 8;
        let bit_idx = i % 8;
        let sign_bit = (packed_signs[byte_idx] >> bit_idx) & 1;
        let sign = if sign_bit == 1 { 1.0f32 } else { -1.0f32 };
        acc += sketched_q[i] * sign;
    }
    acc
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::sketch::QJLSketch;
    use rand::SeedableRng;
    use rand_chacha::ChaCha20Rng;
    use rand_distr::{Distribution, StandardNormal};

    fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
        let normal: StandardNormal = StandardNormal;
        (0..d)
            .map(|_| {
                let v: f64 = normal.sample(rng);
                v as f32
            })
            .collect()
    }

    #[test]
    fn test_signed_dot_all_positive() {
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let packed = vec![0xFF]; // all bits set = all +1
        let result = signed_dot(&q, &packed, 8);
        assert!((result - 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_signed_dot_all_negative() {
        let q = vec![1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let packed = vec![0x00]; // all bits clear = all -1
        let result = signed_dot(&q, &packed, 8);
        assert!((result + 36.0).abs() < 1e-6);
    }

    #[test]
    fn test_score_identical_vectors() {
        let d = 64;
        let s = 512;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let v = random_vec(d, &mut rng);

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&v, 1, &outlier_indices).unwrap();
        let scores = sketch.score(&v, &compressed).unwrap();

        let exact = v.iter().map(|x| x * x).sum::<f32>();
        let relative_error = (scores[0] - exact).abs() / exact;
        assert!(
            relative_error < 0.35,
            "relative error {relative_error} too high (exact={exact}, approx={})",
            scores[0]
        );
    }

    #[test]
    fn test_score_sign_preserved() {
        let d = 64;
        let s = 512;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let q = random_vec(d, &mut rng);
        let k = random_vec(d, &mut rng);
        let exact: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&k, 1, &outlier_indices).unwrap();
        let scores = sketch.score(&q, &compressed).unwrap();

        assert_eq!(
            scores[0] > 0.0,
            exact > 0.0,
            "sign mismatch: approx={}, exact={exact}",
            scores[0]
        );
    }

    #[test]
    fn test_score_multiple_vectors() {
        let d = 16;
        let s = 128;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(77);

        let q = random_vec(d, &mut rng);
        let num_keys = 10;
        let keys: Vec<f32> = (0..num_keys)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, num_keys, &outlier_indices).unwrap();
        let scores = sketch.score(&q, &compressed).unwrap();

        assert_eq!(scores.len(), num_keys);
        for s in &scores {
            assert!(s.is_finite());
        }
    }
}
