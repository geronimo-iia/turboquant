use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

pub fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let normal: StandardNormal = StandardNormal;
    (0..d)
        .map(|_| {
            let v: f64 = normal.sample(rng);
            v as f32
        })
        .collect()
}

pub fn random_vec_with_outliers(
    d: usize,
    outlier_dims: &[usize],
    outlier_scale: f32,
    rng: &mut ChaCha20Rng,
) -> Vec<f32> {
    let mut v = random_vec(d, rng);
    for &dim in outlier_dims {
        v[dim] *= outlier_scale;
    }
    v
}

pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

pub fn argsort_desc(values: &[f32]) -> Vec<usize> {
    let mut indices: Vec<usize> = (0..values.len()).collect();
    indices.sort_by(|&a, &b| {
        values[b]
            .partial_cmp(&values[a])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices
}

pub fn top_k_indices(values: &[f32], k: usize) -> Vec<usize> {
    let sorted = argsort_desc(values);
    sorted.into_iter().take(k).collect()
}

pub fn recall(exact_top_k: &[usize], approx_top_k: &[usize]) -> f32 {
    let exact_set: std::collections::HashSet<usize> = exact_top_k.iter().copied().collect();
    let matches = approx_top_k
        .iter()
        .filter(|i| exact_set.contains(i))
        .count();
    matches as f32 / exact_top_k.len() as f32
}

/// Kendall's tau rank correlation between two rankings.
/// Both inputs are permutations (argsort results).
pub fn kendall_tau(rank_a: &[usize], rank_b: &[usize]) -> f32 {
    let n = rank_a.len();
    assert_eq!(n, rank_b.len());

    // Convert to rank arrays: rank_of[item] = position
    let mut pos_a = vec![0usize; n];
    let mut pos_b = vec![0usize; n];
    for (pos, &item) in rank_a.iter().enumerate() {
        pos_a[item] = pos;
    }
    for (pos, &item) in rank_b.iter().enumerate() {
        pos_b[item] = pos;
    }

    let mut concordant = 0i64;
    let mut discordant = 0i64;
    for i in 0..n {
        for j in (i + 1)..n {
            let a_diff = (pos_a[i] as i64) - (pos_a[j] as i64);
            let b_diff = (pos_b[i] as i64) - (pos_b[j] as i64);
            let product = a_diff * b_diff;
            if product > 0 {
                concordant += 1;
            } else if product < 0 {
                discordant += 1;
            }
        }
    }

    let pairs = (n * (n - 1)) / 2;
    (concordant - discordant) as f32 / pairs as f32
}
