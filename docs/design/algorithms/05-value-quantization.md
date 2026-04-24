# Algorithm 5: Value Quantization

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
