/// Detect outlier dimensions within a group of vectors.
///
/// For each dimension, computes the L2 norm across all vectors in the group.
/// Returns the indices of the top-k dimensions with the highest norms.
///
/// - `keys`: flattened [group_size, head_dim] row-major
/// - `group_size`: number of vectors in the group
/// - `head_dim`: dimension of each vector
/// - `count`: number of outlier dimensions to select
pub fn detect_outliers(keys: &[f32], group_size: usize, head_dim: usize, count: usize) -> Vec<u8> {
    assert_eq!(keys.len(), group_size * head_dim);
    assert!(count <= head_dim);
    assert!(head_dim <= 256, "head_dim must fit in u8");

    // L2 norm per dimension across the group
    let mut dim_norms = vec![0.0f32; head_dim];
    for t in 0..group_size {
        let row = &keys[t * head_dim..(t + 1) * head_dim];
        for (d, &val) in row.iter().enumerate() {
            dim_norms[d] += val * val;
        }
    }
    for n in &mut dim_norms {
        *n = n.sqrt();
    }

    // Top-k by norm (partial sort)
    let mut indices: Vec<u8> = (0..head_dim as u8).collect();
    indices.select_nth_unstable_by(count.saturating_sub(1), |&a, &b| {
        dim_norms[b as usize]
            .partial_cmp(&dim_norms[a as usize])
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    indices.truncate(count);
    indices.sort_unstable();
    indices
}

/// Build an outlier mask from indices. mask[i] = true if i is an outlier.
pub fn outlier_mask(indices: &[u8], head_dim: usize) -> Vec<bool> {
    let mut mask = vec![false; head_dim];
    for &idx in indices {
        mask[idx as usize] = true;
    }
    mask
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_outlier_known_spike() {
        let head_dim = 4;
        let group_size = 8;
        let mut keys = vec![0.1f32; group_size * head_dim];
        // Make dimension 2 a massive outlier
        for t in 0..group_size {
            keys[t * head_dim + 2] = 100.0;
        }
        let indices = detect_outliers(&keys, group_size, head_dim, 1);
        assert_eq!(indices, vec![2]);
    }

    #[test]
    fn test_outlier_count_respected() {
        let head_dim = 8;
        let group_size = 4;
        let keys = vec![1.0f32; group_size * head_dim];
        let indices = detect_outliers(&keys, group_size, head_dim, 3);
        assert_eq!(indices.len(), 3);
    }

    #[test]
    fn test_outlier_multiple() {
        let head_dim = 4;
        let group_size = 4;
        let mut keys = vec![0.1f32; group_size * head_dim];
        // Dims 1 and 3 are outliers
        for t in 0..group_size {
            keys[t * head_dim + 1] = 50.0;
            keys[t * head_dim + 3] = 80.0;
        }
        let indices = detect_outliers(&keys, group_size, head_dim, 2);
        assert!(indices.contains(&1));
        assert!(indices.contains(&3));
    }

    #[test]
    fn test_outlier_mask() {
        let mask = outlier_mask(&[1, 3], 5);
        assert_eq!(mask, vec![false, true, false, true, false]);
    }
}
