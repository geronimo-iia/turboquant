# Algorithm 6: Fused Dequant + Dot Product

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
