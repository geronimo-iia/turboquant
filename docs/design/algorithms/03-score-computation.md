# Algorithm 3: Score Computation

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

## Scale factor: `sqrt(π/2) / sketch_dim`

This comes from the QJL unbiased estimator. The `sqrt(d)` scaling
in the projection rows cancels between query and key (both projected
through the same matrix). The `sqrt(π/2)` corrects the bias introduced
by taking signs.

## Outlier subtraction: `q_inlier = q_full - q_outlier`

The CUDA kernel computes `q_sketch_val = shared_q_sketch - shared_q_outliers_sketch`
for the inlier inner product. This ensures the inlier score only
reflects the inlier dimensions of both query and key.
