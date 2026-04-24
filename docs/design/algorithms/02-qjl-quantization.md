# Algorithm 2: QJL Quantization (sign-based hashing)

Source: `quantize_with_outliers_kernel` in `qjl_quant_kernel.cu`
Impl: `src/quantize.rs` → `QJLSketch::quantize`

Compresses key vectors into packed sign bits. Outlier and inlier
dimensions are projected separately through the same matrix but
stored in separate bit arrays.

```
Input:
  keys            : [num_vectors, d] f32  flattened row-major
  outlier_indices : [outlier_count] u8
  proj_dir_quant  : [s, d] f32

Output:
  key_quant         : [num_vectors, s/8] u8    — inlier sign bits
  key_outlier_quant : [num_vectors, os/8] u8   — outlier sign bits
  key_norms         : [num_vectors] f32        — full vector L2 norm
  outlier_norms     : [num_vectors] f32        — outlier component L2 norm

Algorithm:
  outlier_mask[i] = true if i in outlier_indices

  For each vector x:
    1. key_norms = ||x||
    2. outlier_norms = sqrt(Σ x[i]² for i in outlier_indices)
    3. For each projection p in 0..s:
         dot_inlier  = Σ proj_dir_quant[p, i] * x[i]  where !outlier_mask[i]
         dot_outlier = Σ proj_dir_quant[p, i] * x[i]  where  outlier_mask[i]
    4. Pack sign(dot_inlier) into key_quant, 8 bits per byte
    5. Pack sign(dot_outlier) into key_outlier_quant (first os projections)
```

Bit-packing: bit `i%8` of byte `i/8`. Bit set = positive projection.
