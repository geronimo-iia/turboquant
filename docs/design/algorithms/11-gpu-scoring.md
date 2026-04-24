# Algorithm 11: GPU-Accelerated Scoring

Impl: `src/gpu/mod.rs`, `src/gpu/wgpu_backend.rs`, `src/gpu/score.wgsl`

GPU-accelerated compressed-vs-compressed scoring using WGPU compute
shaders. Transparent runtime dispatch — the public API is unchanged.

## WGSL Compute Shader

```
Kernel: score_compressed_gpu
Workgroup size: 64 threads
Each thread: one vector pair

Bindings (7 total):
  0: params (uniform)     — sketch_dim, outlier_sketch_dim, word counts, num_pairs
  1: a_inlier (storage)   — packed u32 sign bits
  2: b_inlier (storage)
  3: a_outlier (storage)
  4: b_outlier (storage)
  5: norms (storage)      — packed [a_norm, a_out_norm, b_norm, b_out_norm] × num_pairs
  6: scores (storage, rw) — output

Per thread:
  1. XOR inlier u32 words → countOneBits → sum matching bits
  2. sim = matching / sketch_dim
  3. cos_inlier = cos(π × (1 - sim))
  4. Compute inlier norms from full/outlier norms
  5. Same for outlier component
  6. score = inlier_norm_a × inlier_norm_b × cos_inlier
           + outlier_norm_a × outlier_norm_b × cos_outlier
```

## Runtime Dispatch

```rust
score_compressed(a, b):
  if gpu feature enabled AND gpu adapter available AND num_vectors >= gpu_min_batch_compressed():
    → GPU path (WGPU Hamming cosine shader)
  else:
    → CPU path (byte XOR + popcount)

score(query, compressed):
  if gpu feature enabled AND gpu adapter available AND num_vectors >= gpu_min_batch():
    → GPU path (WGPU float×sign shader) [future]
  else:
    → CPU path (signed_dot loop)
```

Two separate thresholds:

| Variable | Default | Path |
|----------|---------|------|
| `QJL_GPU_MIN_BATCH` | 5000 | Float×sign `score()` |
| `QJL_GPU_MIN_BATCH_COMPRESSED` | 100000 | Compressed `score_compressed()` |

Compressed scoring is ~17 ns/vector on CPU — GPU only wins above ~100K
vectors. Float×sign is ~0.6 µs/vector — GPU wins above ~3K-5K vectors.

Override at runtime:

```bash
QJL_GPU_MIN_BATCH=0 cargo run                  # force GPU for float×sign
QJL_GPU_MIN_BATCH_COMPRESSED=0 cargo run        # force GPU for compressed
```

Values are read once from the environment and cached.

## Store-Level Batch Scoring

```rust
KeyStore::score_all_pages(query, sketch, outlier_indices):
  1. Compress query via sketch.quantize
  2. For each page in store:
       score_compressed_pair(query, 0, page_keys, j) for each vector j
  3. Return Vec<(slug_hash, Vec<f32>)>
```

## Feature Flag

```toml
[features]
gpu = ["dep:wgpu", "dep:pollster"]
```

Opt-in at compile time. Default builds have zero GPU dependencies.
GPU tests are `#[ignore]` — run with `cargo test --features gpu -- --ignored`.
