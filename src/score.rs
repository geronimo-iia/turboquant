use crate::quantize::{pack_signs, CompressedKeys};
use crate::sketch::{l2_norm, matvec, QJLSketch};

impl QJLSketch {
    /// Compute approximate attention scores between a query and compressed keys.
    ///
    /// Returns one score per compressed key vector.
    /// Score ≈ dot(query, key) estimated via sign-based Hamming distance.
    ///
    /// - `query`: [head_dim] f32
    /// - `compressed`: compressed key vectors from `quantize()`
    pub fn score(&self, query: &[f32], compressed: &CompressedKeys) -> Vec<f32> {
        let d = self.head_dim;
        let s = self.sketch_dim;
        let os = self.outlier_sketch_dim;
        assert_eq!(query.len(), d);

        let inlier_bytes = s / 8;
        let outlier_bytes = os / 8;
        let q_norm = l2_norm(query);

        // Sketch the query: sketched_q = proj_dir_quant @ query  → [s]
        // (equivalent to query @ proj_dir_score since quant = score^T)
        let sketched_q = matvec(&self.proj_dir_quant, s, d, query);

        // Pack query signs for inlier comparison
        let q_signs_inlier: Vec<bool> = sketched_q.iter().map(|&v| v > 0.0).collect();
        let q_packed_inlier = pack_signs(&q_signs_inlier);

        // Pack query signs for outlier comparison (first os projections)
        let q_signs_outlier: Vec<bool> = sketched_q[..os].iter().map(|&v| v > 0.0).collect();
        let q_packed_outlier = pack_signs(&q_signs_outlier);

        let mut scores = vec![0.0f32; compressed.num_vectors];

        for (v, score) in scores.iter_mut().enumerate().take(compressed.num_vectors) {
            // Inlier score via Hamming distance
            let k_inlier = &compressed.key_quant[v * inlier_bytes..(v + 1) * inlier_bytes];
            let hamming_inlier = hamming_match_count(&q_packed_inlier, k_inlier, s);
            let cos_inlier = estimate_cosine(hamming_inlier, s);

            // Inlier norm = sqrt(full_norm^2 - outlier_norm^2)
            let full_sq = compressed.key_norms[v] * compressed.key_norms[v];
            let outlier_sq = compressed.outlier_norms[v] * compressed.outlier_norms[v];
            let inlier_norm = (full_sq - outlier_sq).max(0.0).sqrt();

            let score_inlier = q_norm * inlier_norm * cos_inlier;

            // Outlier score via Hamming distance
            let k_outlier =
                &compressed.key_outlier_quant[v * outlier_bytes..(v + 1) * outlier_bytes];
            let hamming_outlier = hamming_match_count(&q_packed_outlier, k_outlier, os);
            let cos_outlier = estimate_cosine(hamming_outlier, os);

            let score_outlier = q_norm * compressed.outlier_norms[v] * cos_outlier;

            *score = score_inlier + score_outlier;
        }

        scores
    }
}

/// Count matching sign bits between two packed byte arrays.
/// Returns the number of positions where both signs agree.
fn hamming_match_count(a: &[u8], b: &[u8], total_bits: usize) -> usize {
    let num_bytes = total_bits.div_ceil(8);
    let mut matches = 0usize;
    for i in 0..num_bytes {
        // XOR gives 1 where bits differ, NOT-XOR gives 1 where they match
        let xor = a[i] ^ b[i];
        matches += (!xor).count_ones() as usize;
    }
    // Correct for padding bits in the last byte
    let padding = num_bytes * 8 - total_bits;
    matches - padding
}

/// Estimate cosine similarity from the fraction of matching sign bits.
///
/// cos(θ) ≈ cos(π * (1 - match_fraction))
/// where match_fraction = matching_bits / total_bits
fn estimate_cosine(matching_bits: usize, total_bits: usize) -> f32 {
    let match_frac = matching_bits as f32 / total_bits as f32;
    (std::f32::consts::PI * (1.0 - match_frac)).cos()
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
    fn test_hamming_match_identical() {
        let a = vec![0b10110101u8, 0b11001010];
        let matches = hamming_match_count(&a, &a, 16);
        assert_eq!(matches, 16);
    }

    #[test]
    fn test_hamming_match_opposite() {
        let a = vec![0b11111111u8];
        let b = vec![0b00000000u8];
        let matches = hamming_match_count(&a, &b, 8);
        assert_eq!(matches, 0);
    }

    #[test]
    fn test_estimate_cosine_identical() {
        // All bits match → cos(0) = 1.0
        let cos = estimate_cosine(256, 256);
        assert!((cos - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_estimate_cosine_orthogonal() {
        // Half bits match → cos(π/2) ≈ 0
        let cos = estimate_cosine(128, 256);
        assert!(cos.abs() < 1e-6);
    }

    #[test]
    fn test_score_identical_vectors() {
        let d = 32;
        let s = 256;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let v = random_vec(d, &mut rng);

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&v, 1, &outlier_indices);
        let scores = sketch.score(&v, &compressed);

        let exact = v.iter().map(|x| x * x).sum::<f32>();
        let relative_error = (scores[0] - exact).abs() / exact;
        assert!(
            relative_error < 0.15,
            "relative error {relative_error} too high (exact={exact}, approx={})",
            scores[0]
        );
    }

    #[test]
    fn test_score_sign_preserved() {
        let d = 32;
        let s = 256;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(123);

        let q = random_vec(d, &mut rng);
        let k = random_vec(d, &mut rng);
        let exact: f32 = q.iter().zip(k.iter()).map(|(a, b)| a * b).sum();

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&k, 1, &outlier_indices);
        let scores = sketch.score(&q, &compressed);

        // Sign should match
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
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut rng = ChaCha20Rng::seed_from_u64(77);

        let q = random_vec(d, &mut rng);
        let num_keys = 10;
        let keys: Vec<f32> = (0..num_keys)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, num_keys, &outlier_indices);
        let scores = sketch.score(&q, &compressed);

        assert_eq!(scores.len(), num_keys);
        // All scores should be finite
        for s in &scores {
            assert!(s.is_finite());
        }
    }
}
