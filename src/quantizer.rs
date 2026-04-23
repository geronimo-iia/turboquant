use crate::error::{QjlError, Result};
use crate::outliers::detect_outliers;
use crate::quantize::CompressedKeys;
use crate::sketch::QJLSketch;

/// Streaming key quantizer — accumulates vectors and compresses in batches.
pub struct KeyQuantizer<'a> {
    sketch: &'a QJLSketch,
    outlier_count: usize,
    buffer_size: usize,
    group_size: usize,
    residual: Vec<f32>,
    groups: Vec<CompressedKeys>,
    pub seq_len: usize,
}

impl<'a> KeyQuantizer<'a> {
    pub fn new(
        sketch: &'a QJLSketch,
        outlier_count: usize,
        buffer_size: usize,
        group_size: usize,
    ) -> Result<Self> {
        if !buffer_size.is_multiple_of(group_size) {
            return Err(QjlError::DimensionMismatch {
                expected: buffer_size.next_multiple_of(group_size),
                got: buffer_size,
            });
        }
        Ok(Self {
            sketch,
            outlier_count,
            buffer_size,
            group_size,
            residual: Vec::new(),
            groups: Vec::new(),
            seq_len: 0,
        })
    }

    pub fn build_sketch(&mut self, keys: &[f32], num_vectors: usize) -> Result<()> {
        let d = self.sketch.head_dim;
        if keys.len() != num_vectors * d {
            return Err(QjlError::DimensionMismatch {
                expected: num_vectors * d,
                got: keys.len(),
            });
        }

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
                detect_outliers(group_keys, self.group_size, d, self.outlier_count)?;
            let compressed = self
                .sketch
                .quantize(group_keys, self.group_size, &outlier_indices)?;
            self.groups.push(compressed);
        }

        if remainder > 0 {
            let start = full_groups * self.group_size * d;
            self.residual.extend_from_slice(&keys[start..]);
        }
        Ok(())
    }

    pub fn update(&mut self, key: &[f32]) -> Result<()> {
        let d = self.sketch.head_dim;
        if key.len() != d {
            return Err(QjlError::DimensionMismatch {
                expected: d,
                got: key.len(),
            });
        }

        self.residual.extend_from_slice(key);
        self.seq_len += 1;

        if self.residual.len() < self.buffer_size * d {
            return Ok(());
        }

        let num_vectors = self.buffer_size;
        let full_groups = num_vectors / self.group_size;

        for g in 0..full_groups {
            let start = g * self.group_size * d;
            let end = start + self.group_size * d;
            let group_keys = &self.residual[start..end];

            let outlier_indices =
                detect_outliers(group_keys, self.group_size, d, self.outlier_count)?;
            let compressed = self
                .sketch
                .quantize(group_keys, self.group_size, &outlier_indices)?;
            self.groups.push(compressed);
        }

        self.residual.clear();
        Ok(())
    }

    pub fn attention_score(&self, query: &[f32]) -> Result<Vec<f32>> {
        let d = self.sketch.head_dim;
        if query.len() != d {
            return Err(QjlError::DimensionMismatch {
                expected: d,
                got: query.len(),
            });
        }

        let mut scores = Vec::with_capacity(self.seq_len);

        for group in &self.groups {
            let group_scores = self.sketch.score(query, group)?;
            scores.extend_from_slice(&group_scores);
        }

        let residual_vecs = self.residual.len() / d;
        for v in 0..residual_vecs {
            let vec_data = &self.residual[v * d..(v + 1) * d];
            let dot: f32 = query.iter().zip(vec_data.iter()).map(|(a, b)| a * b).sum();
            scores.push(dot);
        }

        Ok(scores)
    }

    pub fn compressed_len(&self) -> usize {
        self.groups.iter().map(|g| g.num_vectors).sum()
    }

    pub fn residual_len(&self) -> usize {
        self.residual.len() / self.sketch.head_dim
    }

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
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(16, d, &mut rng);
        quantizer.build_sketch(&keys, 16).unwrap();

        assert_eq!(quantizer.seq_len, 16);
        assert_eq!(quantizer.compressed_len(), 16);
        assert!(quantizer.residual_is_empty());
    }

    #[test]
    fn test_build_sketch_with_remainder() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(10, d, &mut rng);
        quantizer.build_sketch(&keys, 10).unwrap();

        assert_eq!(quantizer.seq_len, 10);
        assert_eq!(quantizer.compressed_len(), 8);
        assert_eq!(quantizer.residual_len(), 2);
    }

    #[test]
    fn test_stream_residual_buffer() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..5 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key).unwrap();
        }

        assert_eq!(quantizer.seq_len, 5);
        assert_eq!(quantizer.residual_len(), 5);
        assert_eq!(quantizer.compressed_len(), 0);
    }

    #[test]
    fn test_stream_buffer_flush() {
        let d = 16;
        let s = 64;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..8 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key).unwrap();
        }

        assert_eq!(quantizer.seq_len, 8);
        assert_eq!(quantizer.compressed_len(), 8);
        assert!(quantizer.residual_is_empty());
    }

    #[test]
    fn test_stream_matches_batch() {
        let d = 16;
        let s = 128;
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_keys(16, d, &mut rng);
        let query = random_vec(d, &mut ChaCha20Rng::seed_from_u64(999));

        let mut batch_q = KeyQuantizer::new(&sketch, 2, 16, 8).unwrap();
        batch_q.build_sketch(&keys, 16).unwrap();
        let batch_scores = batch_q.attention_score(&query).unwrap();

        let mut stream_q = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();
        for i in 0..16 {
            stream_q.update(&keys[i * d..(i + 1) * d]).unwrap();
        }
        let stream_scores = stream_q.attention_score(&query).unwrap();

        assert_eq!(batch_scores.len(), stream_scores.len());
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
        let sketch = QJLSketch::new(d, s, s, 42).unwrap();
        let mut quantizer = KeyQuantizer::new(&sketch, 2, 8, 8).unwrap();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        for _ in 0..12 {
            let key = random_vec(d, &mut rng);
            quantizer.update(&key).unwrap();
        }

        let query = random_vec(d, &mut rng);
        let scores = quantizer.attention_score(&query).unwrap();

        assert_eq!(scores.len(), 12);
    }
}
