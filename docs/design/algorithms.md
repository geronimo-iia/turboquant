# TurboQuant — Algorithms

Reference: `amirzandieh/QJL` (Python/CUDA), paper arXiv:2504.19874.
Verified against `qjl_score_kernel.cu` and `qjl_quant_kernel.cu`.

## Architecture

```
src/
├── sketch.rs       QJLSketch — projection matrix, quantize, score
├── outliers.rs     detect_outliers — top-k norms per dimension
├── quantize.rs     CompressedKeys — sign hashing, bit-packing
├── score.rs        score — signed dot, outlier sketch subtraction
├── values.rs       CompressedValues — min-max quantization, fused matmul
├── quantizer.rs    KeyQuantizer — batch + streaming wrapper
├── math.rs         Numerical helpers — lgamma, beta_pdf, normal_icdf, Simpson's rule
└── codebook.rs     Codebook — Lloyd-Max optimal scalar quantization
```

## Algorithm 1: Random Projection Matrix

Source: `QJLSketch.__init__` + `init_rot_dir` in `llama3_utils_qjl.py`
Impl: `src/sketch.rs` → `QJLSketch::new`

Generates an orthogonalized random projection matrix. The QR step
ensures near-orthogonal rows, and the `√d` scaling makes the
sign-based estimator's bias correction work correctly.

```
Input:
  d    = head dimension (e.g. 128)
  s    = sketch dimension, must be divisible by 8
  seed = RNG seed

Output:
  proj_dir_score : [d, s] f32  row-major — orthogonalized, scaled by √d
  proj_dir_quant : [s, d] f32  row-major — transpose of proj_dir_score

Algorithm:
  1. Generate Gaussian random matrix [d, s] (column-major for nalgebra)
  2. num_chunks = ceil(s / d)
  3. For each chunk of d columns:
       Q, R = qr(chunk)
       chunk = Q * sqrt(d)
  4. Convert to row-major [d, s] → proj_dir_score
  5. proj_dir_quant = transpose(proj_dir_score)
```

Both matrices are stored as `Vec<f32>` in row-major order. The
projection `proj_dir_quant @ v` is equivalent to `v @ proj_dir_score`.

## Algorithm 2: QJL Quantization (sign-based hashing)

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

## Algorithm 3: Score Computation

Source: `calc_score_kernel` in `qjl_score_kernel.cu`
Impl: `src/score.rs` → `QJLSketch::score`

Computes approximate `dot(query, key)` from compressed sign bits.
This is the critical algorithm — verified against the CUDA kernel.

```
Input:
  query             : [d] f32
  compressed keys   : from Algorithm 2
  proj_dir_score    : [d, s] f32  row-major
  proj_dir_quant    : [s, d] f32  row-major

Output:
  scores : [num_vectors] f32

Algorithm:
  1. Full query sketch:
       q_sketch = proj_dir_quant @ query                    → [s]

  2. Outlier query sketch (only outlier dims of query projected):
       q_outlier_sketch[p] = Σ query[j] * proj_dir_score[j, p]
                             for j in outlier_indices        → [s]

  3. Inlier query sketch:
       q_inlier_sketch = q_sketch - q_outlier_sketch        → [s]

  4. For each compressed key vector v:
       dot_inlier  = Σ q_inlier_sketch[i]  * sign(key_quant[v, i])     for i in 0..s
       dot_outlier = Σ q_outlier_sketch[i] * sign(key_outlier_quant[v, i])  for i in 0..os

       inlier_norm = sqrt(key_norms[v]² - outlier_norms[v]²)

       score = sqrt(π/2)/s  * inlier_norm       * dot_inlier
             + sqrt(π/2)/os * outlier_norms[v]   * dot_outlier
```

The `signed_dot` function: for each bit position, if bit is set
multiply by +1, else by -1. No Hamming distance, no cosine — the
float query sketch values are used directly.

**Scale factor: `sqrt(π/2) / sketch_dim`**

This comes from the QJL unbiased estimator. The `sqrt(d)` scaling
in the projection rows cancels between query and key (both projected
through the same matrix). The `sqrt(π/2)` corrects the bias introduced
by taking signs.

**Outlier subtraction: `q_inlier = q_full - q_outlier`**

The CUDA kernel computes `q_sketch_val = shared_q_sketch - shared_q_outliers_sketch`
for the inlier inner product. This ensures the inlier score only
reflects the inlier dimensions of both query and key.

## Algorithm 4: Outlier Detection

Source: `QJLKeyQuantizer.build_sketch` in `llama3_utils_qjl.py`
Impl: `src/outliers.rs` → `detect_outliers`

```
Input:
  keys          : [group_size, d] f32  flattened row-major
  outlier_count : usize

Output:
  outlier_indices : [outlier_count] u8  sorted ascending

Algorithm:
  1. For each dimension i in 0..d:
       dim_norm[i] = sqrt(Σ keys[t, i]² for t in 0..group_size)
  2. outlier_indices = top_k(dim_norm, outlier_count)
  3. Sort ascending
```

The norm is computed across the group (not per-vector). Dimensions
with high energy across the group are outliers.

## Algorithm 5: Value Quantization

Source: `triton_quantize_and_pack_along_last_dim` in `new_pack.py`
Impl: `src/values.rs` → `quantize_values`

```
Input:
  values     : [num_elements] f32
  group_size : usize
  bits       : 2 or 4

Output:
  packed : [num_elements / feat_per_int] i32   where feat_per_int = 32/bits
  scale  : [num_groups] f32
  mn     : [num_groups] f32

Algorithm:
  For each group:
    1. mn = min(group), mx = max(group)
    2. scale = (mx - mn) / (2^bits - 1)
    3. quantized[i] = round((value[i] - mn) / scale), clamped to [0, 2^bits-1]
  Pack feat_per_int values per i32:
    packed[i/fpi] |= quantized[i] << ((i % fpi) * bits)
```

## Algorithm 6: Fused Dequant + Dot Product

Source: `cuda_quantized_bmm_dynamic` in `matmul.py`
Impl: `src/values.rs` → `quantized_dot`

```
Input:
  weights    : [num_elements] f32
  compressed : CompressedValues

Output:
  scalar f32

Algorithm:
  acc = 0
  For each element i:
    q_val = (packed[i/fpi] >> ((i%fpi) * bits)) & mask
    float_val = q_val * scale[i/group_size] + mn[i/group_size]
    acc += weights[i] * float_val
```

## Algorithm 7: Lloyd-Max Codebook Generation

Source: Lloyd 1982 / Max 1960 (textbook optimal scalar quantization).
Impl: `src/codebook.rs` → `generate_codebook`, `src/math.rs` → helpers.

Generates an optimal scalar quantization codebook for the coordinate
marginal distribution of a d-dimensional unit-sphere-uniform vector.

```
Input:
  dim        = vector dimension (determines the marginal distribution)
  bit_width  = bits per scalar (1..=8, giving 2^bit_width levels)
  iterations = max Lloyd-Max iterations (typically 50–100)

Output:
  centroids  : [2^bit_width] f32  — reconstruction values
  boundaries : [2^bit_width - 1] f32  — decision thresholds

Algorithm:
  1. Initialize centroids via quantile spacing of Beta(1/2, (d-1)/2):
       c_i = F⁻¹((i + 0.5) / k)  for i in 0..k, where k = 2^bit_width
  2. Lloyd-Max iterations (until convergence or max iterations):
     a. Update boundaries as midpoints:
          b_i = (c_i + c_{i+1}) / 2
     b. Update centroids as conditional means:
          c_i = E[X | b_{i-1} ≤ X < b_i]
        computed via Simpson's rule numerical integration
     c. Stop when max |c_new - c_old| < 1e-12
  3. Final boundaries recomputed from converged centroids

Marginal PDF:
  f(x) = C_d · (1 - x²)^((d-3)/2)  for |x| < 1, d ≤ 50
  f(x) ≈ N(0, 1/d)                   for d > 50
```

Computation is done in f64 for numerical stability; the returned
codebook stores f32 centroids and boundaries.

`CodebookCache` memoizes codebooks by (dim, bit_width) to avoid
recomputation.

## Algorithm 8: Streaming Quantizer

Source: `QJLKeyQuantizer.update_sketch` in `llama3_utils_qjl.py`
Impl: `src/quantizer.rs` → `KeyQuantizer`

```
State:
  groups   : Vec<CompressedKeys>   — compressed groups so far
  residual : Vec<f32>              — uncompressed tail (< buffer_size vectors)
  seq_len  : usize

build_sketch(keys, num_vectors):
  Split into groups of group_size. Compress each group (Algorithms 4+2).
  Remainder goes to residual.

update(key):
  Append to residual. If residual reaches buffer_size:
    Split into groups, compress, append to groups, clear residual.

attention_score(query):
  For compressed groups: use Algorithm 3 (approximate scores).
  For residual vectors: exact dot product (not yet compressed).
  Concatenate and return.
```

## Data Structures (as implemented)

```rust
// src/sketch.rs
pub struct QJLSketch {
    pub head_dim: usize,
    pub sketch_dim: usize,
    pub outlier_sketch_dim: usize,
    pub proj_dir_score: Vec<f32>,     // [head_dim, sketch_dim] row-major
    pub proj_dir_quant: Vec<f32>,     // [sketch_dim, head_dim] row-major
}

// src/quantize.rs
pub struct CompressedKeys {
    pub key_quant: Vec<u8>,           // [num_vectors, sketch_dim/8]
    pub key_outlier_quant: Vec<u8>,   // [num_vectors, outlier_sketch_dim/8]
    pub key_norms: Vec<f32>,          // [num_vectors]
    pub outlier_norms: Vec<f32>,      // [num_vectors]
    pub outlier_indices: Vec<u8>,     // [outlier_count] per group
    pub num_vectors: usize,
    pub head_dim: usize,
}

// src/codebook.rs
pub struct Codebook {
    pub centroids: Vec<f32>,          // [2^bit_width] reconstruction values
    pub boundaries: Vec<f32>,         // [2^bit_width - 1] decision thresholds
    pub bit_width: u8,                // 1..=8
}

// src/values.rs
pub struct CompressedValues {
    pub packed: Vec<i32>,             // bit-packed quantized values
    pub scale: Vec<f32>,              // [num_groups]
    pub mn: Vec<f32>,                 // [num_groups]
    pub num_elements: usize,
    pub bits: u8,                     // 2 or 4
    pub group_size: usize,
}
```

## Quality Characteristics (measured)

Tested with random Gaussian vectors, d=64–128, s=64–512.

| Metric | Value | Conditions |
|--------|-------|------------|
| Distortion (MSE/signal) | < 0.35 | s = 2d |
| Distortion monotonicity | d > 2d > 4d | confirmed |
| Top-10 recall | ≥ 0.55 mean | 200 keys, s = 4d, 100 trials |
| Kendall's tau | > 0.70 mean | 100 keys, s = 4d, 50 trials |
| Outlier benefit | ≥ 20% distortion reduction | 10x outlier magnitude |
| Value 4-bit relative error | < 0.20 mean | random Gaussian |
| Value 2-bit relative error | < 1.0 mean | random Gaussian |

Quality improves with larger sketch_dim. At s = 8d (the paper's
recommended setting), distortion drops significantly and ranking
preservation approaches exact.

## Dependencies

```toml
nalgebra = "0.33"       # QR decomposition
rand = "0.8"            # RNG
rand_chacha = "0.3"     # deterministic seeded RNG
rand_distr = "0.4"      # Normal distribution
bytemuck = "1.16"       # zero-copy casts
rayon = "1.10"          # parallelism (Phase 5)
memmap2 = "0.9"         # persistence (Phase 3)
blake3 = "1.5"          # content hashing (Phase 3)
```
