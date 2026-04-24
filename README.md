# qjl-sketch

TurboQuant vector compression in Rust — sign-based hashing with
near-optimal distortion rate.

Compresses high-dimensional vectors into packed sign bits and scores
queries directly against the compressed representation. No
decompression, no LLM, no GPU required.

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025)
and [QJL](https://github.com/amirzandieh/QJL) (Zandieh et al., 2024).

## Features

- **QJL sign-based key compression** — outlier/inlier separation,
  packed 1 bit per projection
- **Compressed-vs-compressed scoring** — Hamming cosine estimator
  for page-to-page similarity (no decompression needed)
- **Lloyd-Max codebook** — optimal scalar quantization for
  unit-sphere coordinate marginals (1–8 bit)
- **MSE-optimal quantization** — random rotation + Lloyd-Max
  per-coordinate (TurboQuant Stage 1)
- **Min-max value quantization** — 2-bit or 4-bit with i32 bit-packing
- **Score without decompressing** — signed dot product with
  `sqrt(π/2)/s` scale factor, matches the QJL CUDA kernel
- **Streaming quantizer** — batch or one-vector-at-a-time compression
- **Append-only persistence** — mmap-based KeyStore and ValueStore
  with zero-copy page views, streaming export/import
- **Crash recovery** — truncated tail detection, index rebuild
- **Compaction** — reclaim dead space with atomic rename
- **Optional serde** — `Serialize`/`Deserialize` on all public structs
  (feature-gated behind `serde`)

## Quick Start

```rust
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::outliers::detect_outliers;

// Create a sketch (deterministic from seed)
let sketch = QJLSketch::new(128, 256, 64, 42)?;

// Compress key vectors
let outlier_indices = detect_outliers(&keys, group_size, 128, 4)?;
let compressed = sketch.quantize(&keys, num_vectors, &outlier_indices)?;

// Score a query against compressed keys
let scores = sketch.score(&query, &compressed)?;
```

### Persistence

```rust
use qjl_sketch::store::config::KeysConfig;
use qjl_sketch::store::key_store::KeyStore;

let config = KeysConfig { head_dim: 128, sketch_dim: 256,
                           outlier_sketch_dim: 64, seed: 42 };

// Create and populate
let mut store = KeyStore::create(dir, config)?;
store.append(slug_hash, content_hash, &compressed)?;

// Reopen later — sketch reconstructed from seed
let store = KeyStore::open(dir)?;
let page = store.get_page(slug_hash).unwrap();
let reloaded = page.to_compressed_keys(128);
let scores = store.config.build_sketch().score(&query, &reloaded)?;
```

## Quality

Measured over 10K+ random vector pairs (d=64–128, s=64–512).

| Metric                  | Value                      |
| ----------------------- | -------------------------- |
| Distortion (MSE/signal) | < 0.35 at s=2d             |
| Top-10 recall           | ≥ 0.55 mean                |
| Kendall's tau           | > 0.70 mean                |
| Outlier benefit         | ≥ 20% distortion reduction |
| Score persistence       | bit-exact after round-trip |

Quality improves with larger sketch_dim.

## Documentation

- [docs/](docs/README.md) — full documentation index
- [docs/design/](docs/design/README.md) — architecture and design decisions
- [docs/benchmarks.md](docs/benchmarks.md) — benchmark suite and results
- [docs/roadmap.md](docs/roadmap.md) — development roadmap
- [docs/decisions/](docs/decisions/) — architecture decision records


## Performance

Apple M3 Max, d=128, s=256, 32 vectors/page.

| Operation                          | Time    |
| ---------------------------------- | ------- |
| Score 1 page (64 vec)              | 24 us   |
| Score 100 pages                    | 1.8 ms  |
| Score 1000 pages                   | 17.8 ms |
| Key quantize (per vector)          | 38 us   |
| Value quantize 4-bit (per element) | 2.7 ns  |
| Cold start (100 pages)             | 217 us  |
| Page lookup                        | 5 ns    |

### GPU store scoring (`--features gpu`)

`score_all_pages` batches all vectors into a single GPU dispatch.
Without GPU, falls back to `sketch.score()` per page.

| Pages | Vectors | CPU (d=128) | GPU (d=128) | Speedup |
| ----- | ------- | ----------- | ----------- | ------- |
| 100   | 3,200   | 2.2 ms      | 2.3 ms *    | 1x      |
| 1000  | 32,000  | 21.8 ms     | 3.3 ms      | **6.6x** |
| 10000 | 320,000 | 225.8 ms    | 15.5 ms     | **14.6x** |

\* Below `QJL_GPU_MIN_BATCH` (5K vectors), auto-dispatch uses CPU.
Higher dimensions benefit more: d=64 sees 7x at 10K pages, d=128 sees 14.6x.

GPU is beneficial when:
1. Total vectors across all pages >= 5000 for d=64 (default threshold) or pages >= 3000 for d=128
2. Higher vector dimensions (d=128+) see larger speedups
3. The store has many pages (1000+)

GPU is NOT beneficial when:
- Few pages (< ~150 pages at 32 vec/page = 4800 vectors)
- The auto-dispatch correctly falls back to CPU in these cases

See [docs/benchmarks.md](docs/benchmarks.md) for full results.


## Examples

| Example                                                | Command                                                    | What it shows                                                     |
| ------------------------------------------------------ | ---------------------------------------------------------- | ----------------------------------------------------------------- |
| [basic_qjl](examples/basic_qjl.rs)                     | `cargo run --example basic_qjl`                            | Compress vectors, score queries, compare with exact dot products  |
| [compressed_scoring](examples/compressed_scoring.rs)   | `cargo run --example compressed_scoring`                   | Page-to-page similarity via Hamming cosine on sign bits           |
| [mse_quantization](examples/mse_quantization.rs)       | `cargo run --example mse_quantization`                     | Rotation + Lloyd-Max quantization, 2-bit vs 4-bit MSE comparison  |
| [serde_roundtrip](examples/serde_roundtrip.rs)         | `cargo run --example serde_roundtrip --features serde`     | Serialize/deserialize Codebook, QJLSketch, RandomRotation to JSON |
| [store_export_import](examples/store_export_import.rs) | `cargo run --example store_export_import --features serde` | Streaming JSONL export from one KeyStore, import into another     |

## Building

```bash
cargo build                           # debug
cargo test                            # all tests (default features)
cargo test --features serde           # include serde round-trip tests
cargo test --features gpu -- --ignored  # GPU tests (requires adapter)
cargo bench                           # criterion benchmarks (release)
cargo bench --bench gpu_score --features gpu  # GPU vs CPU benchmarks
cargo clippy -- -D warnings           # lint
cargo run --example serde_roundtrip --features serde
cargo run --example store_export_import --features serde
```

Requires Rust 1.95+.

### Feature flags

| Flag    | Default | What it enables                                                         |
| ------- | ------- | ----------------------------------------------------------------------- |
| `serde` | off     | `Serialize`/`Deserialize` on public structs, store export/import        |
| `gpu`   | off     | WGPU compute shaders for GPU-accelerated scoring (auto-detects adapter) |

### Environment variables

| Variable            | Default | What it controls                                                           |
| ------------------- | ------- | -------------------------------------------------------------------------- |
| `QJL_GPU_MIN_BATCH` | 5000    | Total vectors for `score_all_pages` GPU dispatch. Set to `0` to force GPU. |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
