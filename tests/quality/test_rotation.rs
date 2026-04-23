use super::helpers::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use turboquant::sketch::{l2_norm, matvec, QJLSketch};

#[test]
fn test_rotation_preserves_norm() {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, s, 42);
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    let trials = 1000;
    let mut ratio_sum = 0.0f64;

    // proj_dir_quant is [s, d], each row scaled by sqrt(d).
    // ||proj @ v||² = sum_i (proj_i · v)² ≈ s * d * ||v||² / d = s * ||v||²
    // (each projection has variance d * ||v||²/d = ||v||² due to orthogonality)
    // So ||proj @ v|| ≈ sqrt(s) * ||v||
    let expected_scale = (s as f64).sqrt();

    for _ in 0..trials {
        let v = random_vec(d, &mut rng);
        let projected = matvec(&sketch.proj_dir_quant, s, d, &v);
        let v_norm = l2_norm(&v) as f64;
        let p_norm = l2_norm(&projected) as f64;
        let ratio = p_norm / (v_norm * expected_scale);
        ratio_sum += ratio;
    }

    let mean_ratio = ratio_sum / trials as f64;
    assert!(
        (0.90..=1.10).contains(&mean_ratio),
        "mean norm ratio {mean_ratio:.4} outside [0.90, 1.10]"
    );
}

#[test]
fn test_rotation_preserves_inner_product() {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, s, 42);
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    let trials = 1000;
    let mut relative_errors = Vec::with_capacity(trials);

    // proj_q · proj_k ≈ s * (q · k)
    // (each of s projections contributes ≈ q·k on average)
    for _ in 0..trials {
        let q = random_vec(d, &mut rng);
        let k = random_vec(d, &mut rng);

        let exact = dot(&q, &k) as f64;
        if exact.abs() < 1e-3 {
            continue;
        }

        let proj_q = matvec(&sketch.proj_dir_quant, s, d, &q);
        let proj_k = matvec(&sketch.proj_dir_quant, s, d, &k);
        let projected = dot(&proj_q, &proj_k) as f64;

        let normalized = projected / s as f64;
        let rel_err = (normalized - exact).abs() / exact.abs();
        relative_errors.push(rel_err);
    }

    let mean_err: f64 = relative_errors.iter().sum::<f64>() / relative_errors.len() as f64;
    assert!(mean_err < 0.15, "mean relative error {mean_err:.4} >= 0.15");
}
