# TurboQuant — Algorithms for Rust Implementation

Reference: `amirzandieh/QJL` (Python/CUDA), paper arXiv:2504.19874

## Architecture Overview

```
                    ┌─────────────────────────────┐
                    │  turboquant (Rust crate)     │
                    │                              │
                    │  ┌───────────────────────┐   │
                    │  │ QJLSketch             │   │
                    │  │  - proj_dir: [d, s]   │   │
                    │  │  - quantize()         │   │
                    │  │  - score()            │   │
                    │  └───────────────────────┘   │
                    │  ┌───────────────────────┐   │
                    │  │ KeyQuantizer          │   │
                    │  │  - build_sketch()     │   │
                    │  │  - update_sketch()    │   │
                    │  │  - attention_score()  │   │
                    │  └───────────────────────┘   │
                    │  ┌───────────────────────┐   │
                    │  │ ValueQuantizer        │   │
                    │  │  - quantize_pack()    │   │
                    │  │  - dequant_matmul()   │   │
                    │  └───────────────────────┘   │
                    └─────────────────────────────┘
```

## Algorithm 1: Random Projection Matrix (QJLSketch init)

Source: `QJLSketch.__init__` + `init_rot_dir` in `llama3_utils_qjl.py`

**Purpose:** Generate the random projection matrix S used for sign-based
quantization. Orthogonalize it for better numerical properties.

```
Input:
  d     = head dimension (e.g. 128)
  s     = sketch dimension (number of random projections)
  seed  = RNG seed for reproducibility

Output:
  proj_dir       : [d, s] f32  — raw random Gaussian matrix
  proj_dir_score : [d, s] f16  — orthogonalized, scaled by √d

Algorithm:
  1. proj_dir = randn(d, s, seed)           // Gaussian random
  2. num_chunks = ceil(s / d)
  3. for i in 0..num_chunks:
       chunk = proj_dir[:, i*d .. (i+1)*d]  // [d, d] or smaller
       Q, _ = qr(chunk)                     // QR decomposition
       proj_dir_score[:, i*d .. (i+1)*d] = Q * sqrt(d)
  4. proj_dir_quant = transpose(proj_dir_score)  // [s, d]
```

**Rust crates needed:** `ndarray` or raw `Vec<f32>`, `rand` + `rand_distr::Normal`,
QR decomposition from `nalgebra` or `faer`.

## Algorithm 2: QJL Quantization (sign-based hashing)

Source: `QJLSketch.quantize` → CUDA kernel `quantize_with_outliers_kernel`

**Purpose:** Compress key vectors into sign bits. Separate outlier
dimensions for higher-fidelity treatment.

```
Input:
  key_states      : [batch, heads, groups, group_size, d] f32
  outlier_indices : [batch, heads, groups, outlier_count] u8
  proj_dir_quant  : [s, d] f16
  outlier_sketch_dim : int

Output:
  key_quant         : [batch, heads, groups, group_size, s/8] u8  — packed sign bits (inliers)
  key_outlier_quant : [batch, heads, groups, group_size, outlier_sketch_dim/8] u8  — packed sign bits (outliers)
  outlier_norms     : [batch, heads, groups, group_size] f32  — L2 norm of outlier dims

Algorithm:
  for each vector x in key_states[b, h, g, :, :]:
    1. Split x into inlier and outlier components:
         outlier_mask[i] = 1 if i in outlier_indices[b,h,g] else 0
         x_inlier  = x * (1 - outlier_mask)
         x_outlier = x * outlier_mask

    2. Project both:
         sketch_inlier  = proj_dir_quant @ x_inlier    // [s]
         sketch_outlier = proj_dir_quant @ x_outlier    // [outlier_sketch_dim]

    3. Extract signs and bit-pack:
         for each group of 8 projections:
           byte = 0
           for bit in 0..8:
             if sketch_inlier[group*8 + bit] > 0:
               byte |= (1 << bit)
           key_quant[...] = byte

         // Same for outlier sketch

    4. outlier_norms = sqrt(sum(x_outlier^2))
```

**Key insight:** The sign of the random projection preserves inner product
direction (Johnson-Lindenstrauss). Bit-packing 8 signs into 1 byte gives
8x compression on the sign bits themselves.

## Algorithm 3: QJL Score Computation (attention scores from compressed keys)

Source: `QJLSketch.calc_score` → CUDA kernel `qjl_gqa_score_kernel`

**Purpose:** Compute approximate Q·K^T attention scores directly from
compressed sign bits, without decompressing K.

```
Input:
  query_states      : [batch, heads, 1, d] f32
  key_quant         : [batch, heads, groups, group_size, s/8] u8
  key_outlier_quant : [batch, heads, groups, group_size, outlier_s/8] u8
  key_norms         : [batch, heads, groups, group_size] f32
  outlier_norms     : [batch, heads, groups, group_size] f32
  outlier_indices   : [batch, heads, groups, outlier_count] u8
  proj_dir_score    : [d, s] f16

Output:
  scores : [batch, heads, 1, total_seq_len] f32

Algorithm:
  1. Sketch the query:
       sketched_q = query @ proj_dir_score    // [batch, heads, 1, s]

  2. For each compressed key vector:
       // Unpack sign bits from key_quant
       // Count matching signs between sketched_q and key_quant
       // This is a Hamming distance computation:

       sign_match_count = popcount(~(sign_bits_q XOR sign_bits_k))
       // where sign_bits_q = pack(sketched_q > 0)

       cos_estimate = cos(π * hamming_distance / sketch_dim)

       score_inlier = ||q|| * ||k_inlier|| * cos_estimate

  3. Same for outlier component, then:
       score = score_inlier + score_outlier
```

**Key insight:** The score computation is a Hamming distance on packed
bytes — extremely fast with `popcount` intrinsics. No float decompression
needed.

## Algorithm 4: Outlier Detection

Source: `QJLKeyQuantizer.build_sketch` in `llama3_utils_qjl.py`

**Purpose:** Identify which dimensions are outliers within each group.

```
Input:
  key_states    : [batch, heads, groups, group_size, d] f32
  outlier_count : int (e.g. 7)

Output:
  outlier_indices : [batch, heads, groups, outlier_count] u8

Algorithm:
  for each group [b, h, g]:
    1. norms = ||key_states[b,h,g,:,:]||  along group_size dim  // [d]
       (L2 norm across the group_size vectors for each dimension)
    2. outlier_indices = top_k(norms, outlier_count)
```

## Algorithm 5: Value Quantization (min-max + bit-packing)

Source: `triton_quantize_and_pack_along_last_dim` in `new_pack.py`

**Purpose:** Quantize V matrices with simple min-max scalar quantization
and pack into int32.

```
Input:
  value_states : [batch, heads, d, seq_len] f32
  group_size   : int
  bits         : int (2 or 4)

Output:
  packed : [batch, heads, d, seq_len / (32/bits)] i32
  scale  : [batch, heads, d, num_groups] f32
  mn     : [batch, heads, d, num_groups] f32

Algorithm:
  num_groups = seq_len / group_size
  feat_per_int = 32 / bits

  for each group:
    1. mn = min(group), mx = max(group)
    2. scale = (mx - mn) / (2^bits - 1)
    3. quantized = round((value - mn) / scale)  // clamp to [0, 2^bits-1]
    4. Pack `feat_per_int` quantized values into one i32:
         packed = 0
         for i in 0..feat_per_int:
           packed |= quantized[i] << (i * bits)
```

## Algorithm 6: Quantized Value MatMul

Source: `cuda_quantized_bmm_dynamic` in `matmul.py`

**Purpose:** Compute attention_weights @ V without fully decompressing V.

```
Input:
  attn_weights : [batch, heads, 1, seq_len] f32
  packed_v     : [batch, heads, seq_len/(32/bits), d] i32
  scale        : [batch, heads, num_groups, d] f32
  mn           : [batch, heads, num_groups, d] f32
  bits         : int

Output:
  result : [batch, heads, 1, d] f32

Algorithm:
  Fused dequant + matmul:
  for each output dimension j:
    acc = 0
    for each group g:
      for each element i in group:
        // Extract quantized value from packed int
        shift = (i % feat_per_int) * bits
        mask = (1 << bits) - 1
        q_val = (packed[...] >> shift) & mask
        // Dequantize
        float_val = q_val * scale[g, j] + mn[g, j]
        // Accumulate
        acc += attn_weights[..., g*group_size + i] * float_val
    result[j] = acc
```

## Algorithm 7: Streaming Update (online quantization)

Source: `QJLKeyQuantizer.update_sketch` in `llama3_utils_qjl.py`

**Purpose:** Append new key vectors to the compressed store one at a time
(for autoregressive decoding or streaming ingest).

```
Input:
  new_key : [batch, heads, 1, d] f32

State:
  residual_buffer : accumulates until buffer_size is reached
  key_quant, key_outlier_quant, key_norms, outlier_norms, outlier_indices

Algorithm:
  1. Append new_key to residual_buffer
  2. If len(residual_buffer) < buffer_size: return (not enough to quantize)
  3. Reshape buffer into groups of group_size
  4. Detect outliers (Algorithm 4)
  5. Quantize (Algorithm 2)
  6. Compute norms
  7. Concatenate with existing compressed state
  8. Clear residual_buffer
```

## Data Structures Summary

```rust
struct QJLSketch {
    dim: (usize, usize),          // (head_dim, sketch_dim)
    outlier_sketch_dim: usize,
    proj_dir_quant: Vec<f32>,     // [sketch_dim, head_dim] row-major
    proj_dir_score: Vec<f32>,     // [head_dim, sketch_dim] row-major
}

struct CompressedKeys {
    key_quant: Vec<u8>,           // packed sign bits (inliers)
    key_outlier_quant: Vec<u8>,   // packed sign bits (outliers)
    key_norms: Vec<f32>,          // per-vector L2 norms
    outlier_norms: Vec<f32>,      // per-vector outlier L2 norms
    outlier_indices: Vec<u8>,     // per-group outlier dim indices
    residual: Option<Vec<f32>>,   // un-quantized tail (< buffer_size)
    seq_len: usize,
}

struct CompressedValues {
    packed: Vec<i32>,             // bit-packed quantized values
    scale: Vec<f32>,              // per-group scale
    mn: Vec<f32>,                 // per-group minimum
    full_tail: Vec<f32>,          // un-quantized tail (< buffer_size)
    bits: u8,                     // 2 or 4
    group_size: usize,
}
```

## Rust Implementation Plan

### Phase 1 — CPU-only, f32, no SIMD

1. `QJLSketch::new(head_dim, sketch_dim, seed)` — Algorithms 1
2. `QJLSketch::quantize(keys, outlier_indices)` — Algorithm 2 (loop-based)
3. `QJLSketch::score(query, compressed_keys)` — Algorithm 3 (popcount via `u8::count_ones()`)
4. `detect_outliers(keys, count)` — Algorithm 4
5. `quantize_values(values, group_size, bits)` — Algorithm 5
6. `quantized_matmul(weights, compressed_values)` — Algorithm 6
7. `KeyQuantizer` with streaming update — Algorithm 7

### Phase 2 — SIMD + performance

- Use `std::arch` for `_mm256_popcnt_epi8` (AVX-512 VPOPCNT) or
  `popcnt` on u64 chunks
- Batch the projection as a GEMM via `ndarray` + BLAS
- Parallelize with `rayon` for multi-head processing

### Phase 3 — GPU (optional)

- Port CUDA kernels to `wgpu` compute shaders or use `cudarc` for
  direct CUDA FFI
- The CUDA kernels in `QJL/qjl_kernel/csrc/` are the reference

## Crate Dependencies

```toml
[dependencies]
ndarray = "0.16"           # matrix ops
nalgebra = "0.33"          # QR decomposition
rand = "0.8"               # RNG
rand_distr = "0.4"         # Normal distribution
rayon = "1.10"             # parallelism
```
