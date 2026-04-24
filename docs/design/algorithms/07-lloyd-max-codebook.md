# Algorithm 7: Lloyd-Max Codebook Generation

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
