// Lloyd-Max codebook generation for scalar quantization of
// unit-sphere coordinate marginals.
//
// Inspired by turboquant (MIT, https://github.com/abdelstark/turboquant).
// Algorithm: Lloyd 1982 / Max 1960 — textbook optimal scalar quantization.

use std::collections::HashMap;

use crate::error::{QjlError, Result};
use crate::math::{beta_pdf, sample_beta_marginal, simpson_integrate};

/// A Lloyd-Max optimal scalar quantization codebook.
///
/// Stores 2^bit_width centroids (reconstruction values) and
/// 2^bit_width - 1 decision boundaries, optimized for the Beta(1/2, (d-1)/2)
/// coordinate marginal of unit-sphere-uniform vectors.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Quantization centroids (reconstruction values), length = 2^bit_width.
    pub centroids: Vec<f32>,
    /// Decision boundaries between adjacent centroids, length = 2^bit_width - 1.
    pub boundaries: Vec<f32>,
    /// Number of bits per scalar.
    pub bit_width: u8,
}

impl Codebook {
    /// Number of quantization levels.
    pub fn num_levels(&self) -> usize {
        self.centroids.len()
    }

    /// Quantize a scalar value to its codebook index via binary search.
    pub fn quantize(&self, value: f32) -> u8 {
        if value.is_nan() {
            return 0;
        }
        let v = value as f64;
        match self.boundaries.binary_search_by(|b| {
            (*b as f64)
                .partial_cmp(&v)
                .unwrap_or(std::cmp::Ordering::Equal)
        }) {
            Ok(i) => i as u8,
            Err(i) => i.min(self.centroids.len() - 1) as u8,
        }
    }

    /// Reconstruct the centroid value for a codebook index.
    pub fn dequantize(&self, index: u8) -> f32 {
        self.centroids[index as usize]
    }
}

/// Generate a Lloyd-Max optimal codebook for dimension `dim` with
/// `bit_width` bits per scalar (1..=8) using up to `iterations` refinement steps.
///
/// Computation is done in f64 for numerical stability; the returned
/// codebook stores f32 centroids and boundaries.
pub fn generate_codebook(dim: usize, bit_width: u8, iterations: usize) -> Result<Codebook> {
    if !(1..=8).contains(&bit_width) {
        return Err(QjlError::InvalidCodebookBitWidth(bit_width));
    }
    if dim == 0 {
        return Err(QjlError::InvalidDimension(dim));
    }

    let k = 1usize << bit_width;

    // Initialize centroids via quantile spacing of the marginal distribution.
    let mut centroids: Vec<f64> = (0..k)
        .map(|i| sample_beta_marginal(dim, (i as f64 + 0.5) / k as f64))
        .collect();
    centroids.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));

    for _ in 0..iterations {
        let boundaries: Vec<f64> = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

        let new_centroids = compute_centroids(&centroids, &boundaries, dim);

        let converged = centroids
            .iter()
            .zip(new_centroids.iter())
            .all(|(a, b)| (a - b).abs() < 1e-12);

        centroids = new_centroids;
        if converged {
            break;
        }
    }

    let boundaries: Vec<f64> = centroids.windows(2).map(|w| (w[0] + w[1]) / 2.0).collect();

    Ok(Codebook {
        centroids: centroids.iter().map(|&c| c as f32).collect(),
        boundaries: boundaries.iter().map(|&b| b as f32).collect(),
        bit_width,
    })
}

/// Compute conditional-mean centroids E[X | region] for each Voronoi cell.
fn compute_centroids(old: &[f64], boundaries: &[f64], dim: usize) -> Vec<f64> {
    let k = old.len();
    let n_points = 200;

    (0..k)
        .map(|i| {
            let lo = if i == 0 { -1.0 } else { boundaries[i - 1] }.max(-0.9999);
            let hi = if i == k - 1 { 1.0 } else { boundaries[i] }.min(0.9999);

            if hi <= lo {
                return old[i];
            }

            let (num, den) = simpson_integrate(lo, hi, n_points, |x| {
                let pdf = beta_pdf(x, dim);
                (x * pdf, pdf)
            });

            if den.abs() < 1e-15 {
                old[i]
            } else {
                num / den
            }
        })
        .collect()
}

/// Memoizing cache for codebooks, keyed by (dim, bit_width).
pub struct CodebookCache {
    books: HashMap<(usize, u8), Codebook>,
}

impl CodebookCache {
    pub fn new() -> Self {
        Self {
            books: HashMap::new(),
        }
    }

    /// Get a cached codebook or generate one (100 Lloyd-Max iterations).
    pub fn get_or_generate(&mut self, dim: usize, bit_width: u8) -> Result<&Codebook> {
        let key = (dim, bit_width);
        if let std::collections::hash_map::Entry::Vacant(e) = self.books.entry(key) {
            e.insert(generate_codebook(dim, bit_width, 100)?);
        }
        Ok(&self.books[&key])
    }
}

impl Default for CodebookCache {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn codebook_1bit_symmetric() {
        let cb = generate_codebook(128, 1, 50).unwrap();
        assert_eq!(cb.centroids.len(), 2);
        assert_eq!(cb.boundaries.len(), 1);
        assert!((cb.centroids[0] + cb.centroids[1]).abs() < 1e-6);
        assert!(cb.boundaries[0].abs() < 1e-6);
    }

    #[test]
    fn codebook_4bit_sorted() {
        let cb = generate_codebook(128, 4, 100).unwrap();
        assert_eq!(cb.centroids.len(), 16);
        assert_eq!(cb.boundaries.len(), 15);
        for w in cb.centroids.windows(2) {
            assert!(w[0] < w[1]);
        }
    }

    #[test]
    fn codebook_centroids_in_range() {
        let cb = generate_codebook(32, 2, 50).unwrap();
        for c in &cb.centroids {
            assert!(*c > -1.0 && *c < 1.0, "centroid {c} out of range");
        }
    }

    #[test]
    fn codebook_boundaries_between_centroids() {
        let cb = generate_codebook(64, 2, 50).unwrap();
        for (i, &b) in cb.boundaries.iter().enumerate() {
            assert!(
                b > cb.centroids[i] && b < cb.centroids[i + 1],
                "boundary {b} not between {} and {}",
                cb.centroids[i],
                cb.centroids[i + 1]
            );
        }
    }

    #[test]
    fn quantize_dequantize_roundtrip() {
        let cb = generate_codebook(64, 3, 50).unwrap();
        for &val in &[-0.5, -0.1, 0.0, 0.1, 0.5] {
            let idx = cb.quantize(val);
            let recon = cb.dequantize(idx);
            assert!(
                (val - recon).abs() < 0.5,
                "val={val}, recon={recon}, idx={idx}"
            );
        }
    }

    #[test]
    fn quantize_extreme_values() {
        let cb = generate_codebook(64, 4, 50).unwrap();
        assert_eq!(cb.quantize(100.0) as usize, cb.centroids.len() - 1);
        assert_eq!(cb.quantize(-100.0), 0);
    }

    #[test]
    fn quantize_nan_returns_zero() {
        let cb = generate_codebook(64, 2, 50).unwrap();
        assert_eq!(cb.quantize(f32::NAN), 0);
    }

    #[test]
    fn codebook_8bit() {
        let cb = generate_codebook(128, 8, 50).unwrap();
        assert_eq!(cb.centroids.len(), 256);
        assert_eq!(cb.boundaries.len(), 255);
    }

    #[test]
    fn codebook_large_dim_gaussian_path() {
        let cb = generate_codebook(256, 4, 100).unwrap();
        assert_eq!(cb.centroids.len(), 16);
        for w in cb.centroids.windows(2) {
            assert!(w[0] < w[1]);
        }
        for c in &cb.centroids {
            assert!(c.abs() < 0.5, "centroid {c} too far from 0 for dim=256");
        }
    }

    #[test]
    fn invalid_bit_width() {
        assert!(generate_codebook(64, 0, 50).is_err());
        assert!(generate_codebook(64, 9, 50).is_err());
    }

    #[test]
    fn invalid_dimension() {
        assert!(generate_codebook(0, 4, 50).is_err());
    }

    #[test]
    fn cache_returns_same_codebook() {
        let mut cache = CodebookCache::new();
        let c1 = cache.get_or_generate(64, 2).unwrap().centroids.clone();
        let c2 = cache.get_or_generate(64, 2).unwrap().centroids.clone();
        assert_eq!(c1, c2);
    }

    #[test]
    fn cache_different_configs() {
        let mut cache = CodebookCache::new();
        let len1 = cache.get_or_generate(64, 2).unwrap().centroids.len();
        let len2 = cache.get_or_generate(64, 4).unwrap().centroids.len();
        assert_ne!(len1, len2);
    }

    #[test]
    fn cache_default() {
        let mut cache = CodebookCache::default();
        assert!(cache.get_or_generate(32, 2).is_ok());
    }

    #[test]
    fn cache_propagates_errors() {
        let mut cache = CodebookCache::new();
        assert!(cache.get_or_generate(0, 2).is_err());
        assert!(cache.get_or_generate(64, 0).is_err());
    }
}
