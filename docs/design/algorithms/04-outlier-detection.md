# Algorithm 4: Outlier Detection

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
