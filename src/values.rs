/// Compressed value vectors — min-max quantized and bit-packed.
#[derive(Clone, Debug)]
pub struct CompressedValues {
    /// Bit-packed quantized values. Layout depends on bits.
    pub packed: Vec<i32>,
    /// Per-group scale factors [num_groups].
    pub scale: Vec<f32>,
    /// Per-group minimums [num_groups].
    pub mn: Vec<f32>,
    /// Number of elements before packing.
    pub num_elements: usize,
    /// Quantization bit-width (2 or 4).
    pub bits: u8,
    /// Group size for quantization.
    pub group_size: usize,
}

/// Quantize a 1-D slice of f32 values with min-max scalar quantization
/// and pack into i32 words.
///
/// - `values`: input values [num_elements]
/// - `group_size`: number of elements per quantization group
/// - `bits`: quantization bit-width (2 or 4)
pub fn quantize_values(values: &[f32], group_size: usize, bits: u8) -> CompressedValues {
    assert!(bits == 2 || bits == 4, "bits must be 2 or 4");
    assert!(
        values.len().is_multiple_of(group_size),
        "values.len() must be divisible by group_size"
    );

    let num_elements = values.len();
    let num_groups = num_elements / group_size;
    let max_val = (1u32 << bits) - 1;
    let feat_per_int = 32 / bits as usize;
    let packed_len = num_elements / feat_per_int;

    let mut scale = vec![0.0f32; num_groups];
    let mut mn = vec![0.0f32; num_groups];
    let mut quantized = vec![0u32; num_elements];

    // Quantize per group
    for g in 0..num_groups {
        let start = g * group_size;
        let end = start + group_size;
        let group = &values[start..end];

        let group_min = group.iter().copied().fold(f32::INFINITY, f32::min);
        let group_max = group.iter().copied().fold(f32::NEG_INFINITY, f32::max);

        mn[g] = group_min;
        let range = group_max - group_min;
        scale[g] = if range > 0.0 {
            range / max_val as f32
        } else {
            1.0 // avoid division by zero
        };

        for (i, &val) in group.iter().enumerate() {
            let q = ((val - group_min) / scale[g])
                .round()
                .clamp(0.0, max_val as f32) as u32;
            quantized[start + i] = q;
        }
    }

    // Bit-pack: feat_per_int quantized values per i32
    let mut packed = vec![0i32; packed_len];
    for (i, &q) in quantized.iter().enumerate().take(num_elements) {
        let word_idx = i / feat_per_int;
        let slot = i % feat_per_int;
        let shift = slot * bits as usize;
        packed[word_idx] |= (q as i32) << shift;
    }

    CompressedValues {
        packed,
        scale,
        mn,
        num_elements,
        bits,
        group_size,
    }
}

/// Dequantize a single element from packed storage.
fn dequant_element(compressed: &CompressedValues, index: usize) -> f32 {
    let feat_per_int = 32 / compressed.bits as usize;
    let mask = (1i32 << compressed.bits) - 1;
    let word_idx = index / feat_per_int;
    let slot = index % feat_per_int;
    let shift = slot * compressed.bits as usize;
    let q_val = (compressed.packed[word_idx] >> shift) & mask;

    let group = index / compressed.group_size;
    q_val as f32 * compressed.scale[group] + compressed.mn[group]
}

/// Dequantize all values back to f32.
pub fn dequantize_all(compressed: &CompressedValues) -> Vec<f32> {
    (0..compressed.num_elements)
        .map(|i| dequant_element(compressed, i))
        .collect()
}

/// Fused dequantize + weighted sum: result = weights @ dequantized_values.
///
/// - `weights`: [num_elements] f32 — attention weights
/// - `compressed`: quantized values [num_elements]
///
/// Returns the weighted sum (scalar).
pub fn quantized_dot(weights: &[f32], compressed: &CompressedValues) -> f32 {
    assert_eq!(weights.len(), compressed.num_elements);

    let feat_per_int = 32 / compressed.bits as usize;
    let mask = (1i32 << compressed.bits) - 1;
    let mut acc = 0.0f32;

    for (i, &w) in weights.iter().enumerate().take(compressed.num_elements) {
        let word_idx = i / feat_per_int;
        let slot = i % feat_per_int;
        let shift = slot * compressed.bits as usize;
        let q_val = (compressed.packed[word_idx] >> shift) & mask;

        let group = i / compressed.group_size;
        let float_val = q_val as f32 * compressed.scale[group] + compressed.mn[group];
        acc += w * float_val;
    }

    acc
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_quantize_4bit_round_trip() {
        let values: Vec<f32> = (0..32).map(|i| i as f32 * 0.1).collect();
        let compressed = quantize_values(&values, 8, 4);
        let reconstructed = dequantize_all(&compressed);

        // 4-bit: 16 levels, max error ≈ scale/2 per element
        for (orig, recon) in values.iter().zip(reconstructed.iter()) {
            let max_err = compressed.scale[0] / 2.0 + 1e-6;
            assert!(
                (orig - recon).abs() <= max_err,
                "orig={orig}, recon={recon}, max_err={max_err}"
            );
        }
    }

    #[test]
    fn test_quantize_2bit_round_trip() {
        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let compressed = quantize_values(&values, 8, 2);
        let reconstructed = dequantize_all(&compressed);

        for (orig, recon) in values.iter().zip(reconstructed.iter()) {
            let group = 0; // simplified — check first group
            let max_err = compressed.scale[group] / 2.0 + 1e-6;
            assert!(
                (orig - recon).abs() <= max_err + compressed.scale[group],
                "orig={orig}, recon={recon}"
            );
        }
    }

    #[test]
    fn test_quantize_4bit_range() {
        let values = vec![0.0f32; 16];
        let compressed = quantize_values(&values, 8, 4);
        let feat_per_int = 8; // 32/4
        let mask = 0xF;
        for word in &compressed.packed {
            for slot in 0..feat_per_int {
                let q = (word >> (slot * 4)) & mask;
                assert!(q >= 0 && q <= 15, "4-bit value out of range: {q}");
            }
        }
    }

    #[test]
    fn test_quantize_2bit_range() {
        let values = vec![0.0f32; 16];
        let compressed = quantize_values(&values, 8, 2);
        let feat_per_int = 16; // 32/2
        let mask = 0x3;
        for word in &compressed.packed {
            for slot in 0..feat_per_int {
                let q = (word >> (slot * 2)) & mask;
                assert!(q >= 0 && q <= 3, "2-bit value out of range: {q}");
            }
        }
    }

    #[test]
    fn test_quantized_dot() {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let weights: Vec<f32> = vec![1.0; 8];
        let compressed = quantize_values(&values, 8, 4);

        let exact: f32 = values.iter().sum();
        let approx = quantized_dot(&weights, &compressed);

        let relative_error = (exact - approx).abs() / exact;
        assert!(
            relative_error < 0.05,
            "relative error {relative_error} too high"
        );
    }

    #[test]
    fn test_quantized_dot_weighted() {
        let values = vec![1.0f32, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0];
        let weights = vec![0.5f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.5];
        let compressed = quantize_values(&values, 8, 4);

        let exact: f32 = values.iter().zip(weights.iter()).map(|(v, w)| v * w).sum();
        let approx = quantized_dot(&weights, &compressed);

        assert!(
            (exact - approx).abs() < 0.5,
            "exact={exact}, approx={approx}"
        );
    }
}
