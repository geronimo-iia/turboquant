use crate::outliers::detect_outliers;
use crate::quantize::CompressedKeys;
use crate::sketch::QJLSketch;

/// Streaming key quantizer — accumulates vectors and compresses in batches.
pub struct KeyQuantizer<'a> {
    sketch: &'a QJLSketch,
    outlier_count: usize,
    buffer_size: usize,
    group_size: usize,
    /// Accumulated uncompressed vectors (residual buffer).
    residual: Vec<f32>,
    /// Compressed groups accumulated so far.
    groups: Vec<CompressedKeys>,
    /// Total sequence length (compressed + residual).
    pub seq_len: usize,
}

impl<'a> KeyQuantizer<'a> {
    pub fn new(
        sketch: &'a QJLSketch,
        outlier_count: usize,
        buffer_size: usize,
        group_size: usize,
    ) -> Self {
        assert!(buffer_size.is_multiple_of(group_size));
        Self {
            sketch,
            outlier_count,
            buffer_size,
            group_size,
            residual: Vec::new(),
            groups: Vec::new(),
            seq_len: 0,
        }
    }

    /// Batch compress a full set of key vectors.
    ///
    /// - `keys`: flattened [num_vectors, head_dim] row-major
    /// - `num_vectors`: total number of vectors
    pub fn build_sketch(&mut self, keys: &[f32], num_vectors: usize) {
        let d = self.sketch.head_dim;
        assert_eq!(keys.len(), num_vectors * d);

        self.groups.clear();
        self.residual.clear();
        self.seq_len = num_vectors;

        let full_groups = num_vectors / self.group_size;
        let remainder = num_vectors % self.group_size;

        for g in 0..full_groups {
            let start = g * self.group_size * d;
            let end = start + self.group_size * d;
            let group_keys = &keys[start..end];

            let outlier_indices =
                detect_outliers(group_keys, self.group_size, d, self.outlier_count);
            let compressed = self
                .sketch
                .quantize(group_keys, self.group_size, &outlier_indices);
            self.groups.push(compressed);
        }

        if remainder > 0 {
            let start = full_groups * self.group_size * d;
            self.residual.extend_from_slice(&keys[start..]);
        }
    }

    /// Append a single vector to the quantizer (streaming mode).
    ///
    /// - `key`: [head_dim] f32
    pub fn update(&mut self, key: &[f32]) {
        let d = self.sketch.head_dim;
        assert_eq!(key.len(), d);

        self.residual.extend_from_slice(key);
        self.seq_len += 1;

        if self.residual.len() < self.buffer_size * d {
            return;
        }

        // Flush buffer: split into groups and compress
        let num_vectors = self.buffer_size;
        let full_groups = num_vectors / self.group_size;

        for g in 0..full_groups {
            let start = g * self.group_size * d;
            let end = start + self.group_size * d;
            let group_keys = &self.residual[start..end];

            let outlier_indices =
                detect_outliers(group_keys, self.group_size, d, self.outlier_count);
            let compressed = self
                .sketch
                .quantize(group_keys, self.group_size, &outlier_indices);
            self.groups.push(compressed);
        }

        self.residual.clear();
    }

    /// Compute attention scores for a query against all compressed + residual keys.
    ///
    /// - `query`: [head_dim] f32
    ///
    /// Returns scores [seq_len].
    pub fn attention_score(&self, query: &[f32]) -> Vec<f32> {
        let d = self.sketch.head_dim;
        assert_eq!(query.len(), d);

        let mut scores = Vec::with_capacity(self.seq_len);

        // Scores from compressed groups
        for group in &self.groups {
            let group_scores = self.sketch.score(query, group);
            scores.extend_from_slice(&group_scores);
        }

        // Scores from residual (exact dot product — not yet compressed)
        let residual_vecs = self.residual.len() / d;
        for v in 0..residual_vecs {
            let vec_data = &self.residual[v * d..(v + 1) * d];
            let dot: f32 = query.iter().zip(vec_data.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot);
        }

        scores
    }

    /// Number of compressed vectors (excluding residual).
    pub fn compressed_len(&self) -> usize {
        self.groups.iter().map(|g| g.num_vectors).sum()
    }

    /// Number of vectors in the residual buffer.
    pub fn residual_len(&self) -> usize {
        self.residual.len() / self.sketch.head_dim
    }

    /// Whether the residual buffer is empty.
    pub fn residual_is_empty(&self) -> bool {
        self.residual.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn random_keys(num: usize, d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
        (0..num).flat_map(|_| random_vec(d, rng)).collect()
    }

    #[test]
    fn test_build_sketch() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(16, d, &mut rng);
        quantizer.build_sketch(&keys, 16);

        assert_eq!(quantizer.seq_len, 16);
        assert_eq!(quantizer.compressed_len(), 16);
        assert!(quantizer.residual_is_empty());
    }

    #[test]
    fn test_build_sketch_with_remainder() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(10, d, &mut rng);
        quantizer.build_sketch(&keys, 10);

        assert_eq!(quantizer.seq_len, 10);
        assert_eq!(quantizer.compressed_len(), 8);
        assert_eq!(quantizer.residual_len(), 2);
    }

    #[test]
    fn test_stream_residual_buffer() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..5 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key);
        }

        assert_eq!(quantizer.seq_len, 5);
        assert_eq!(quantizer.residual_len(), 5);
        assert_eq!(quantizer.compressed_len(), 0);
    }

    #[test]
    fn test_stream_buffer_flush() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..8 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key);
        }

        assert_eq!(quantizer.seq_len, 8);
        assert_eq!(quantizer.compressed_len(), 8);
        assert!(quantizer.residual_is_empty());
    }

    #[test]
    fn test_stream_matches_batch() {
        let d = 16;
        let s = 128;
        let sketch = QJLSketch::new(d, s, s, 42);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(16, d, &mut rng);
        let query = random_vec(d, &mut ChaCha20Rng::seed_from_u64(999));

        // Batch
        let mut batch_q = KeyQuantizer::new(&sketch, 2, 16, 8);
        batch_q.build_sketch(&keys, 16);
        let batch_scores = batch_q.attention_score(&query);

        // Stream
        let mut stream_q = KeyQuantizer::new(&sketch, 2, 8, 8);
        for i in 0..16 {
            stream_q.update(&keys[i * d..(i + 1) * d]);
        }
        let stream_scores = stream_q.attention_score(&query);

        assert_eq!(batch_scores.len(), stream_scores.len());
        // Scores should be close (not identical due to different grouping)
        for (b, s) in batch_scores.iter().zip(stream_scores.iter()) {
            assert!(
                (b - s).abs() < 1.0,
                "batch={b}, stream={s}, diff={}",
                (b - s).abs()
            );
        }
    }

    #[test]
    fn test_attention_score_length() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42);
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8);

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..12 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key);
        }

        let query = random_vec(d, &mut rng);
        let scores = quantizer.attention_score(&query);

        // 8 compressed + 4 residual = 12
        assert_eq!(scores.len(), 12);
    }
}
