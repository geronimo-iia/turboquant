use super::helpers::*;
use qjl_sketch::sketch::QJLSketch;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;

#[test]
fn test_compressed_ranking_preservation() {
    let d = 64;
    let s = 256;
    let num_keys = 100;
    let num_trials = 50;
    let sketch = QJLSketch::new(d, s, s, 42).unwrap();

    let mut tau_sum = 0.0f32;

    for trial in 0..num_trials {
        let mut rng = ChaCha20Rng::seed_from_u64(700 + trial);
        let q = random_vec(d, &mut rng);
        let keys: Vec<f32> = (0..num_keys)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();

        // Exact scores
        let exact_scores: Vec<f32> = (0..num_keys)
            .map(|i| dot(&q, &keys[i * d..(i + 1) * d]))
            .collect();
        let exact_ranking = argsort_desc(&exact_scores);

        // Compressed-vs-compressed scores
        let outlier_indices = vec![0u8];
        let cq = sketch.quantize(&q, 1, &outlier_indices).unwrap();
        let ck = sketch.quantize(&keys, num_keys, &outlier_indices).unwrap();

        let approx_scores: Vec<f32> = (0..num_keys)
            .map(|j| sketch.score_compressed_pair(&cq, 0, &ck, j).unwrap())
            .collect();
        let approx_ranking = argsort_desc(&approx_scores);

        tau_sum += kendall_tau(&exact_ranking, &approx_ranking);
    }

    let mean_tau = tau_sum / num_trials as f32;
    assert!(
        mean_tau > 0.50,
        "mean Kendall's tau {mean_tau:.3} <= 0.50 over {num_trials} trials"
    );
}
