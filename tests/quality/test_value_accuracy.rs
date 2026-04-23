use super::helpers::*;
use qjl_sketch::values::{quantize_values, quantized_dot};
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

fn measure_value_error(bits: u8, trials: usize, seed: u64) -> f64 {
    let group_size = 32;
    let num_elements = 128; // must be divisible by group_size
    let mut rng = ChaCha20Rng::seed_from_u64(seed);

    let mut relative_errors = Vec::with_capacity(trials);

    for _ in 0..trials {
        let values = random_vec(num_elements, &mut rng);
        let weights = random_vec(num_elements, &mut rng);

        let exact: f32 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();

        if exact.abs() < 1.0 {
            continue;
        }

        let compressed = quantize_values(&values, group_size, bits).unwrap();
        let approx = quantized_dot(&weights, &compressed).unwrap();

        let rel_err = ((exact - approx) / exact).abs() as f64;
        relative_errors.push(rel_err);
    }

    relative_errors.iter().sum::<f64>() / relative_errors.len() as f64
}

#[test]
fn test_value_quantized_matmul_error_4bit() {
    let mean_err = measure_value_error(4, 1000, 700);
    assert!(
        mean_err < 0.20,
        "4-bit mean relative error {mean_err:.4} >= 0.20"
    );
}

#[test]
fn test_value_quantized_matmul_error_2bit() {
    let mean_err = measure_value_error(2, 1000, 800);
    assert!(
        mean_err < 1.0,
        "2-bit mean relative error {mean_err:.4} >= 1.0"
    );
}
