# Algorithm 1: Random Projection Matrix

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
