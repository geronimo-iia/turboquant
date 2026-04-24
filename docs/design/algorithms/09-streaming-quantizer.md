# Algorithm 9: Streaming Quantizer

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
