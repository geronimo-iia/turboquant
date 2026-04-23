use super::helpers::*;
use qjl_sketch::sketch::QJLSketch;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn measure_distortion_with_outliers(
    d: usize,
    s: usize,
    outlier_dims: &[usize],
    outlier_scale: f32,
    outlier_count: usize,
    trials: usize,
    seed: u64,
) -> f64 {
    let sketch = QJLSketch::new(d, s, s, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut mse_sum = 0.0f64;
    let mut signal_sum = 0.0f64;

    let outlier_indices: Vec<u8> = if outlier_count > 0 {
        outlier_dims
            .iter()
            .take(outlier_count)
            .map(|&d| d as u8)
            .collect()
    } else {
        vec![]
    };

    for _ in 0..trials {
        let q = random_vec(d, &mut rng);
        let k = random_vec_with_outliers(d, outlier_dims, outlier_scale, &mut rng);

        let exact = dot(&q, &k) as f64;

        let compressed = sketch.quantize(&k, 1, &outlier_indices).unwrap();
        let scores = sketch.score(&q, &compressed).unwrap();
        let approx = scores[0] as f64;

        mse_sum += (exact - approx).powi(2);
        signal_sum += exact.powi(2);
    }

    mse_sum / signal_sum
}

#[test]
fn test_outlier_vs_no_outlier() {
    let d = 128;
    let s = 256;
    let outlier_dims = vec![10, 50];
    let outlier_scale = 10.0;
    let trials = 5000;

    let distortion_no_outlier = measure_distortion_with_outliers(
        d,
        s,
        &outlier_dims,
        outlier_scale,
        0, // no outlier separation
        trials,
        900,
    );

    let distortion_with_outlier = measure_distortion_with_outliers(
        d,
        s,
        &outlier_dims,
        outlier_scale,
        2, // separate the 2 outlier dims
        trials,
        900,
    );

    let improvement = 1.0 - (distortion_with_outlier / distortion_no_outlier);
    assert!(
        improvement >= 0.20,
        "outlier separation improvement {:.1}% < 20% \
         (without={distortion_no_outlier:.4}, with={distortion_with_outlier:.4})",
        improvement * 100.0
    );
}
