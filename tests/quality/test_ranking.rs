use super::helpers::*;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use turboquant::sketch::QJLSketch;

#[test]
fn test_top_k_recall() {
    let d = 64;
    let s = 256;
    let num_keys = 200;
    let k = 10;
    let num_trials = 100;
    let sketch = QJLSketch::new(d, s, s, 42);

    let mut recall_sum = 0.0f32;

    for trial in 0..num_trials {
        let mut rng = ChaCha20Rng::seed_from_u64(500 + trial);
        let q = random_vec(d, &mut rng);
        let keys: Vec<f32> = (0..num_keys)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();

        // Exact scores
        let exact_scores: Vec<f32> = (0..num_keys)
            .map(|i| dot(&q, &keys[i * d..(i + 1) * d]))
            .collect();
        let exact_top = top_k_indices(&exact_scores, k);

        // Compressed scores
        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, num_keys, &outlier_indices);
        let approx_scores = sketch.score(&q, &compressed);
        let approx_top = top_k_indices(&approx_scores, k);

        recall_sum += recall(&exact_top, &approx_top);
    }

    let mean_recall = recall_sum / num_trials as f32;
    assert!(
        mean_recall >= 0.55,
        "mean top-{k} recall {mean_recall:.3} < 0.55 over {num_trials} trials"
    );
}

#[test]
fn test_kendall_tau() {
    let d = 64;
    let s = 256;
    let num_keys = 100;
    let num_trials = 50;
    let sketch = QJLSketch::new(d, s, s, 42);

    let mut tau_sum = 0.0f32;

    for trial in 0..num_trials {
        let mut rng = ChaCha20Rng::seed_from_u64(600 + trial);
        let q = random_vec(d, &mut rng);
        let keys: Vec<f32> = (0..num_keys)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();

        let exact_scores: Vec<f32> = (0..num_keys)
            .map(|i| dot(&q, &keys[i * d..(i + 1) * d]))
            .collect();
        let exact_ranking = argsort_desc(&exact_scores);

        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, num_keys, &outlier_indices);
        let approx_scores = sketch.score(&q, &compressed);
        let approx_ranking = argsort_desc(&approx_scores);

        tau_sum += kendall_tau(&exact_ranking, &approx_ranking);
    }

    let mean_tau = tau_sum / num_trials as f32;
    assert!(
        mean_tau > 0.70,
        "mean Kendall's tau {mean_tau:.3} <= 0.70 over {num_trials} trials"
    );
}
