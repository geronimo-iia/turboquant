use crate::error::{validate_finite, QjlError, Result};
use crate::quantize::CompressedKeys;
use crate::sketch::{matvec, QJLSketch};

/// Hamming similarity between two packed sign bit arrays.
///
/// Returns the fraction of matching bits in \[0.0, 1.0\].
pub fn hamming_similarity(a: &[u8], b: &[u8], total_bits: usize) -> f32 {
    let matching: u32 = a
        .iter()
        .zip(b.iter())
        .map(|(x, y)| (!(x ^ y)).count_ones())
        .sum();
    let padding = (a.len() * 8 - total_bits) as u32;
    (matching - padding) as f32 / total_bits as f32
}

/// Validate that two `CompressedKeys` were produced with compatible sketch parameters.
fn validate_compressed_pair(a: &CompressedKeys, b: &CompressedKeys) -> Result<()> {
    if a.head_dim != b.head_dim {
        return Err(QjlError::SketchParamMismatch {
            context: "head_dim differs",
        });
    }
    let a_inlier_bytes = a.key_quant.len() / a.num_vectors.max(1);
    let b_inlier_bytes = b.key_quant.len() / b.num_vectors.max(1);
    if a_inlier_bytes != b_inlier_bytes {
        return Err(QjlError::SketchParamMismatch {
            context: "sketch_dim differs",
        });
    }
    let a_outlier_bytes = a.key_outlier_quant.len() / a.num_vectors.max(1);
    let b_outlier_bytes = b.key_outlier_quant.len() / b.num_vectors.max(1);
    if a_outlier_bytes != b_outlier_bytes {
        return Err(QjlError::SketchParamMismatch {
            context: "outlier_sketch_dim differs",
        });
    }
    Ok(())
}

/// Compute the compressed-vs-compressed score for a single vector pair.
///
/// Uses Hamming-based cosine estimation on sign bits, weighted by norms.
#[allow(clippy::too_many_arguments)]
fn score_compressed_single(
    a_inlier: &[u8],
    a_outlier: &[u8],
    a_norm: f32,
    a_outlier_norm: f32,
    b_inlier: &[u8],
    b_outlier: &[u8],
    b_norm: f32,
    b_outlier_norm: f32,
    sketch_dim: usize,
    outlier_sketch_dim: usize,
) -> f32 {
    let sim_inlier = hamming_similarity(a_inlier, b_inlier, sketch_dim);
    let cos_inlier = (std::f32::consts::PI * (1.0 - sim_inlier)).cos();
    let inlier_norm_a = (a_norm * a_norm - a_outlier_norm * a_outlier_norm)
        .max(0.0)
        .sqrt();
    let inlier_norm_b = (b_norm * b_norm - b_outlier_norm * b_outlier_norm)
        .max(0.0)
        .sqrt();
    let score_inlier = inlier_norm_a * inlier_norm_b * cos_inlier;

    let score_outlier = if outlier_sketch_dim > 0 {
        let sim_outlier = hamming_similarity(a_outlier, b_outlier, outlier_sketch_dim);
        let cos_outlier = (std::f32::consts::PI * (1.0 - sim_outlier)).cos();
        a_outlier_norm * b_outlier_norm * cos_outlier
    } else {
        0.0
    };

    score_inlier + score_outlier
}

impl QJLSketch {
    /// Compute approximate attention scores between a query and compressed keys.
    ///
    /// Matches the QJL CUDA kernel (`calc_score_kernel`):
    ///   q_sketch = query @ proj_dir_score  (full sketch)
    ///   q_outlier_sketch\[i\] = Σ_j query\[outlier_j\] * proj_dir_score\[outlier_j, i\]
    ///   q_inlier_sketch = q_sketch - q_outlier_sketch
    ///   score = sqrt(π/2)/s * ||k_inlier|| * Σ sign(k_inlier_i) * q_inlier_sketch\[i\]
    ///         + sqrt(π/2)/os * ||k_outlier|| * Σ sign(k_outlier_i) * q_outlier_sketch\[i\]
    ///
    /// - `query`: \[head_dim\] f32
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

        let scl = (std::f32::consts::FRAC_PI_2).sqrt() / s as f32;
        let scl_outlier = (std::f32::consts::FRAC_PI_2).sqrt() / os as f32;

        let mut scores = vec![0.0f32; compressed.num_vectors];

        for (v, score) in scores.iter_mut().enumerate().take(compressed.num_vectors) {
            let k_inlier = &compressed.key_quant[v * inlier_bytes..(v + 1) * inlier_bytes];
            let dot_inlier = signed_dot(&q_inlier_sketch, k_inlier, s);

            let full_sq = compressed.key_norms[v] * compressed.key_norms[v];
            let outlier_sq = compressed.outlier_norms[v] * compressed.outlier_norms[v];
            let inlier_norm = (full_sq - outlier_sq).max(0.0).sqrt();

            let k_outlier =
                &compressed.key_outlier_quant[v * outlier_bytes..(v + 1) * outlier_bytes];
            let dot_outlier = signed_dot(&q_outlier_sketch[..os], k_outlier, os);

            *score = scl * inlier_norm * dot_inlier
                + scl_outlier * compressed.outlier_norms[v] * dot_outlier;
        }

        Ok(scores)
    }

    /// Score two compressed key sets against each other.
    ///
    /// Uses Hamming distance on packed sign bits to estimate the cosine
    /// of the angle between each vector pair, then scales by norms.
    /// Returns one score per vector pair (a\[i\] vs b\[i\]).
    pub fn score_compressed(&self, a: &CompressedKeys, b: &CompressedKeys) -> Result<Vec<f32>> {
        if a.num_vectors != b.num_vectors {
            return Err(QjlError::DimensionMismatch {
                expected: a.num_vectors,
                got: b.num_vectors,
            });
        }
        validate_compressed_pair(a, b)?;

        let s = self.sketch_dim;
        let os = self.outlier_sketch_dim;
        let ib = s / 8;
        let ob = os / 8;

        let scores = (0..a.num_vectors)
            .map(|v| {
                score_compressed_single(
                    &a.key_quant[v * ib..(v + 1) * ib],
                    &a.key_outlier_quant[v * ob..(v + 1) * ob],
                    a.key_norms[v],
                    a.outlier_norms[v],
                    &b.key_quant[v * ib..(v + 1) * ib],
                    &b.key_outlier_quant[v * ob..(v + 1) * ob],
                    b.key_norms[v],
                    b.outlier_norms[v],
                    s,
                    os,
                )
            })
            .collect();

        Ok(scores)
    }

    /// Score a single compressed vector from `a` against one from `b`.
    ///
    /// Extracts vector `i` from `a` and vector `j` from `b`.
    pub fn score_compressed_pair(
        &self,
        a: &CompressedKeys,
        i: usize,
        b: &CompressedKeys,
        j: usize,
    ) -> Result<f32> {
        if i >= a.num_vectors {
            return Err(QjlError::IndexOutOfBounds {
                index: i,
                len: a.num_vectors,
            });
        }
        if j >= b.num_vectors {
            return Err(QjlError::IndexOutOfBounds {
                index: j,
                len: b.num_vectors,
            });
        }
        validate_compressed_pair(a, b)?;

        let s = self.sketch_dim;
        let os = self.outlier_sketch_dim;
        let ib = s / 8;
        let ob = os / 8;

        Ok(score_compressed_single(
            &a.key_quant[i * ib..(i + 1) * ib],
            &a.key_outlier_quant[i * ob..(i + 1) * ob],
            a.key_norms[i],
            a.outlier_norms[i],
            &b.key_quant[j * ib..(j + 1) * ib],
            &b.key_outlier_quant[j * ob..(j + 1) * ob],
            b.key_norms[j],
            b.outlier_norms[j],
            s,
            os,
        ))
    }
}

/// Compute the dot product between float query sketch and packed sign bits.
///
/// For each projection i: result += sketched_q\[i\] * sign(sketched_k\[i\])
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

    // --- signed_dot tests ---

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

    // --- hamming_similarity tests ---

    #[test]
    fn test_hamming_similarity_identical() {
        let a = vec![0xAB, 0xCD];
        assert!((hamming_similarity(&a, &a, 16) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_similarity_opposite() {
        let a = vec![0xFF];
        let b = vec![0x00];
        assert!(hamming_similarity(&a, &b, 8).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_similarity_half() {
        // 0xFF = 11111111, 0x0F = 00001111 → upper 4 differ, lower 4 match → 0.5
        let a = vec![0xFF];
        let b = vec![0x0F];
        assert!((hamming_similarity(&a, &b, 8) - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_hamming_similarity_non_byte_aligned() {
        // 2 bytes = 16 bits, but only 13 bits used. Identical → 1.0
        let a = vec![0xAB, 0x1F];
        assert!((hamming_similarity(&a, &a, 13) - 1.0).abs() < 1e-6);
    }

    // --- score (float query vs compressed) tests ---

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

    // --- score_compressed tests ---

    #[test]
    fn test_score_compressed_identical() {
        let d = 64;
        let s = 512;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let v = random_vec(d, &mut rng);

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&v, 1, &outlier_indices).unwrap();
        let scores = sketch.score_compressed(&compressed, &compressed).unwrap();

        // Same bits vs itself → cos(0) = 1 → score ≈ ||v||²
        let exact = v.iter().map(|x| x * x).sum::<f32>();
        let relative_error = (scores[0] - exact).abs() / exact;
        assert!(
            relative_error < 0.01,
            "relative error {relative_error} too high (exact={exact}, approx={})",
            scores[0]
        );
    }

    #[test]
    fn test_score_compressed_sign_preserved() {
        let d = 64;
        let s = 512;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(300);

        let num_trials = 20;
        let mut correct = 0;
        for _ in 0..num_trials {
            let a_vec = random_vec(d, &mut rng);
            let b_vec = random_vec(d, &mut rng);
            let exact: f32 = a_vec.iter().zip(b_vec.iter()).map(|(x, y)| x * y).sum();

            let outlier_indices = vec![0u8];
            let ca = sketch.quantize(&a_vec, 1, &outlier_indices).unwrap();
            let cb = sketch.quantize(&b_vec, 1, &outlier_indices).unwrap();
            let scores = sketch.score_compressed(&ca, &cb).unwrap();

            if (scores[0] > 0.0) == (exact > 0.0) {
                correct += 1;
            }
        }
        assert!(
            correct >= num_trials * 3 / 4,
            "sign preserved only {correct}/{num_trials} times"
        );
    }

    #[test]
    fn test_score_compressed_dimension_mismatch() {
        let d = 16;
        let s = 32;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let v1 = random_vec(d, &mut rng);
        let v2: Vec<f32> = (0..2).flat_map(|_| random_vec(d, &mut rng)).collect();
        let outlier_indices = vec![0u8];

        let c1 = sketch.quantize(&v1, 1, &outlier_indices).unwrap();
        let c2 = sketch.quantize(&v2, 2, &outlier_indices).unwrap();

        assert!(sketch.score_compressed(&c1, &c2).is_err());
    }

    // --- score_compressed_pair tests ---

    #[test]
    fn test_score_compressed_pair_matches_batch() {
        let d = 32;
        let s = 128;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(500);

        let num = 5;
        let keys: Vec<f32> = (0..num).flat_map(|_| random_vec(d, &mut rng)).collect();
        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, num, &outlier_indices).unwrap();

        let batch = sketch.score_compressed(&compressed, &compressed).unwrap();
        for (i, &batch_score) in batch.iter().enumerate().take(num) {
            let pair = sketch
                .score_compressed_pair(&compressed, i, &compressed, i)
                .unwrap();
            assert!(
                (batch_score - pair).abs() < 1e-6,
                "mismatch at {i}: batch={batch_score}, pair={pair}"
            );
        }
    }

    #[test]
    fn test_score_compressed_pair_cross_index() {
        let d = 32;
        let s = 128;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(600);

        let a_keys: Vec<f32> = (0..3).flat_map(|_| random_vec(d, &mut rng)).collect();
        let b_keys: Vec<f32> = (0..4).flat_map(|_| random_vec(d, &mut rng)).collect();
        let outlier_indices = vec![0u8];
        let ca = sketch.quantize(&a_keys, 3, &outlier_indices).unwrap();
        let cb = sketch.quantize(&b_keys, 4, &outlier_indices).unwrap();

        for i in 0..3 {
            for j in 0..4 {
                let score = sketch.score_compressed_pair(&ca, i, &cb, j).unwrap();
                assert!(score.is_finite(), "non-finite score at ({i}, {j})");
            }
        }
    }

    #[test]
    fn test_score_compressed_pair_out_of_bounds() {
        let d = 16;
        let s = 32;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(42);

        let v = random_vec(d, &mut rng);
        let outlier_indices = vec![0u8];
        let c = sketch.quantize(&v, 1, &outlier_indices).unwrap();

        assert!(sketch.score_compressed_pair(&c, 1, &c, 0).is_err());
        assert!(sketch.score_compressed_pair(&c, 0, &c, 1).is_err());
    }
}
