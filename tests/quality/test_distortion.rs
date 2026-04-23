use super::helpers::*;
use qjl_sketch::sketch::QJLSketch;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn measure_distortion(d: usize, s: usize, trials: usize, seed: u64) -> f64 {
    let sketch = QJLSketch::new(d, s, s, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut mse_sum = 0.0f64;
    let mut signal_sum = 0.0f64;

    for _ in 0..trials {
        let q = random_vec(d, &mut rng);
        let k = random_vec(d, &mut rng);

        let exact = dot(&q, &k) as f64;

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&k, 1, &outlier_indices).unwrap();
        let scores = sketch.score(&q, &compressed).unwrap();
        let approx = scores[0] as f64;

        mse_sum += (exact - approx).powi(2);
        signal_sum += exact.powi(2);
    }

    mse_sum / signal_sum
}

#[test]
fn test_distortion_rate() {
    let d = 128;
    let s = 2 * d; // sketch_dim = 2 * head_dim
    let distortion = measure_distortion(d, s, 10_000, 300);

    assert!(
        distortion < 0.35,
        "distortion {distortion:.4} >= 0.35 at sketch_dim = 2*d"
    );
}

#[test]
fn test_distortion_decreases_with_sketch_dim() {
    let d = 64;
    let trials = 5000;

    let d1 = measure_distortion(d, d, trials, 400);
    let d2 = measure_distortion(d, 2 * d, trials, 400);
    let d4 = measure_distortion(d, 4 * d, trials, 400);

    assert!(
        d1 > d2,
        "distortion at s=d ({d1:.4}) should be > distortion at s=2d ({d2:.4})"
    );
    assert!(
        d2 > d4,
        "distortion at s=2d ({d2:.4}) should be > distortion at s=4d ({d4:.4})"
    );
}
