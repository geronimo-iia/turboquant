# API Overview

Public API surface of `qjl-sketch`, organized by use case.

## Compress

### QJL sign-based compression

```rust
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::outliers::detect_outliers;

let sketch = QJLSketch::new(head_dim, sketch_dim, outlier_sketch_dim, seed)?;
let outliers = detect_outliers(&keys, group_size, head_dim, count)?;
let compressed = sketch.quantize(&keys, num_vectors, &outliers)?;
```

| Item                                           | Module     | Description                                  |
| ---------------------------------------------- | ---------- | -------------------------------------------- |
| `QJLSketch`                                    | `sketch`   | Random projection matrix (QR-orthogonalized) |
| `QJLSketch::new(d, s, os, seed)`               | `sketch`   | Create sketch (deterministic from seed)      |
| `QJLSketch::quantize(keys, n, outliers)`       | `quantize` | Compress vectors to packed sign bits         |
| `CompressedKeys`                               | `quantize` | Packed sign bits + norms + outlier info      |
| `detect_outliers(keys, group, d, count)`       | `outliers` | Top-k dimension norms                        |
| `outlier_mask(indices, d)`                     | `outliers` | Bool mask from outlier indices               |
| `pack_signs(bools)` / `unpack_signs(bytes, n)` | `quantize` | Bit packing utilities                        |

### Value quantization (min-max)

```rust
use qjl_sketch::values::{quantize_values, dequantize_all, quantized_dot};

let compressed = quantize_values(&values, group_size, bits)?;
let reconstructed = dequantize_all(&compressed);
let dot = quantized_dot(&weights, &compressed)?;
```

| Item                                        | Module   | Description                             |
| ------------------------------------------- | -------- | --------------------------------------- |
| `quantize_values(values, group_size, bits)` | `values` | Min-max 2/4-bit quantization            |
| `CompressedValues`                          | `values` | Bit-packed quantized values + scale/min |
| `dequantize_all(compressed)`                | `values` | Reconstruct all values                  |
| `quantized_dot(weights, compressed)`        | `values` | Fused dequant + weighted sum            |

### MSE-optimal quantization (rotation + Lloyd-Max)

```rust
use qjl_sketch::rotation::RandomRotation;
use qjl_sketch::codebook::generate_codebook;
use qjl_sketch::mse_quant::{mse_quantize, mse_dequantize, mse_score};

let rot = RandomRotation::new(dim, seed)?;
let cb = generate_codebook(dim, bit_width, iterations)?;
let quantized = mse_quantize(&vectors, num, &rot, &cb)?;
let reconstructed = mse_dequantize(&quantized, &rot, &cb)?;
let scores = mse_score(&query, &quantized, &rot, &cb)?;
```

| Item                                  | Module      | Description                            |
| ------------------------------------- | ----------- | -------------------------------------- |
| `RandomRotation`                      | `rotation`  | d x d orthogonal matrix (Haar-uniform) |
| `RandomRotation::new(dim, seed)`      | `rotation`  | Create rotation (deterministic)        |
| `RandomRotation::rotate(x)`           | `rotation`  | Apply rotation                         |
| `RandomRotation::rotate_inverse(y)`   | `rotation`  | Apply inverse rotation                 |
| `Codebook`                            | `codebook`  | Lloyd-Max centroids + boundaries       |
| `generate_codebook(dim, bits, iters)` | `codebook`  | Generate optimal codebook (1-8 bit)    |
| `Codebook::quantize(value)`           | `codebook`  | Scalar to codebook index               |
| `Codebook::dequantize(index)`         | `codebook`  | Codebook index to centroid             |
| `CodebookCache`                       | `codebook`  | Memoizing cache keyed by (dim, bits)   |
| `MseQuantized`                        | `mse_quant` | Per-coordinate codebook indices        |
| `mse_quantize(vecs, n, rot, cb)`      | `mse_quant` | Rotate + quantize per coordinate       |
| `mse_dequantize(q, rot, cb)`          | `mse_quant` | Dequantize + inverse rotate            |
| `mse_score(query, q, rot, cb)`        | `mse_quant` | Score query against quantized vectors  |

### Streaming compression

```rust
use qjl_sketch::quantizer::KeyQuantizer;

let mut kq = KeyQuantizer::new(&sketch, group_size, outlier_count, buffer_size)?;
kq.build_sketch(&keys, num_vectors)?;  // batch
kq.update(&single_key)?;               // one at a time
let scores = kq.attention_score(&query)?;
```

| Item           | Module      | Description                           |
| -------------- | ----------- | ------------------------------------- |
| `KeyQuantizer` | `quantizer` | Batch + streaming compression wrapper |

## Score

```rust
// Float x sign (query vs compressed keys)
let scores = sketch.score(&query, &compressed)?;

// Compressed x compressed (page-to-page similarity)
let scores = sketch.score_compressed(&a, &b)?;
let score = sketch.score_compressed_pair(&a, i, &b, j)?;

// Standalone Hamming similarity
let sim = hamming_similarity(&a_bytes, &b_bytes, total_bits);
```

| Item                                           | Module  | Description                                |
| ---------------------------------------------- | ------- | ------------------------------------------ |
| `QJLSketch::score(query, compressed)`          | `score` | Float x sign inner product estimate        |
| `QJLSketch::score_compressed(a, b)`            | `score` | Hamming cosine between two compressed sets |
| `QJLSketch::score_compressed_pair(a, i, b, j)` | `score` | Single pair from different sets            |
| `hamming_similarity(a, b, bits)`               | `score` | Fraction of matching bits [0, 1]           |

## Store

```rust
use qjl_sketch::store::key_store::KeyStore;
use qjl_sketch::store::value_store::ValueStore;
use qjl_sketch::store::config::{KeysConfig, ValuesConfig};

// Create / open
let mut store = KeyStore::create(dir, config)?;
let store = KeyStore::open(dir)?;

// Read / write
store.append(slug_hash, content_hash, &compressed)?;
let page = store.get_page(slug_hash);
let fresh = store.is_fresh(slug_hash, content_hash);

// Maintenance
store.compact()?;

// Score all pages (GPU-accelerated with `gpu` feature)
let results = store.score_all_pages(&query, &sketch, &outliers)?;

// Export / import (requires `serde` feature)
for entry in store.iter_pages() { /* ... */ }
store.import_entry(&entry)?;
```

| Item                                        | Module               | Description                         |
| ------------------------------------------- | -------------------- | ----------------------------------- |
| `KeyStore`                                  | `store::key_store`   | Append-only compressed key storage  |
| `KeyStore::create(dir, config)`             |                      | Create new store                    |
| `KeyStore::open(dir)`                       |                      | Open existing (with crash recovery) |
| `KeyStore::append(slug, hash, keys)`        |                      | Store compressed keys               |
| `KeyStore::get_page(slug)`                  |                      | Zero-copy page lookup               |
| `KeyStore::is_fresh(slug, hash)`            |                      | Check staleness                     |
| `KeyStore::compact()`                       |                      | Reclaim dead space                  |
| `KeyStore::score_all_pages(q, sketch, out)` |                      | Score query against all pages       |
| `KeyStore::iter_pages()`                    |                      | Streaming export (serde)            |
| `KeyStore::import_entry(entry)`             |                      | Streaming import (serde)            |
| `KeyPageView`                               | `store::key_store`   | Zero-copy view into mmap'd page     |
| `KeyExportEntry`                            | `store::key_store`   | Export entry (serde)                |
| `ValueStore`                                | `store::value_store` | Same API pattern for values         |
| `ValueExportEntry`                          | `store::value_store` | Export entry (serde)                |
| `KeysConfig` / `ValuesConfig`               | `store::config`      | Store configuration                 |
| `IndexEntry` / `IndexMeta`                  | `store::config`      | Index structures                    |

## Error handling

```rust
use qjl_sketch::error::{QjlError, Result, validate_finite};
```

All public functions return `Result<T, QjlError>`. Variants:
`DimensionMismatch`, `InvalidSketchDim`, `InvalidBitWidth`,
`NonFiniteInput`, `InvalidCodebookBitWidth`, `InvalidDimension`,
`SketchParamMismatch`, `IndexOutOfBounds`, `StoreMagicMismatch`,
`StoreVersionMismatch`, `OutlierIndexOutOfRange`, `Io`.

## Feature flags

| Flag    | What it enables                                               |
| ------- | ------------------------------------------------------------- |
| `serde` | Serialize/Deserialize on public structs, store export/import  |
| `gpu`   | WGPU GPU-accelerated `score_all_pages` (batched float x sign) |

## Utilities

| Item                               | Module   | Description                                  |
| ---------------------------------- | -------- | -------------------------------------------- |
| `matvec(mat, rows, cols, vec)`     | `sketch` | Matrix-vector multiply                       |
| `l2_norm(v)`                       | `sketch` | Vector L2 norm                               |
| `validate_finite(values, context)` | `error`  | Reject NaN/Inf                               |
| `gpu_min_batch()`                  | `gpu`    | Current GPU dispatch threshold (gpu feature) |
