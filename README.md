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

## Performance

Apple M3 Max, d=128, s=256, 32 vectors/page.

| Operation                          | Time   |
| ---------------------------------- | ------ |
| Score 1 page (float×sign)          | 26 µs  |
| Score 100 pages                    | 1.95 ms |
| Score 1000 pages                   | 19.4 ms |
| Compressed scoring 1000 pages      | 840 µs |
| Key quantize (per vector)          | 43 µs  |
| Value quantize 4-bit (per element) | 2.9 ns |
| Cold start (100 pages)             | 228 µs |
| Page lookup                        | 5 ns   |

### GPU scoring (`--features gpu`)

| Vectors | CPU time | GPU time | GPU per-vector |
| ------- | -------- | -------- | -------------- |
| 100     | 1.77 µs  | 1.72 ms  | 17.2 µs        |
| 1,000   | 17.7 µs  | 1.85 ms  | 1.85 µs       |
| 10,000  | 177 µs   | 2.28 ms  | 0.23 µs       |
| 100,000 | 1.77 ms  | 3.44 ms  | 0.034 µs      |

GPU overhead is ~1.7 ms. CPU compressed scoring is ~17.7 ns/vector.
GPU wins above ~100K vectors on Apple M3 Max.
Run `./scripts/bench.sh --gpu --save` for full results.

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

| Document                                                    | What it covers                        |
| ----------------------------------------------------------- | ------------------------------------- |
| [docs/study.md](docs/study.md)                              | TurboQuant algorithm overview         |
| [docs/design/algorithms/](docs/design/algorithms/README.md) | Algorithm catalog with pseudocode     |
| [docs/design/persistence.md](docs/design/persistence.md)    | Two-store file format                 |
| [docs/design/store.md](docs/design/store.md)                | Store API and lifecycle               |
| [docs/design/serde.md](docs/design/serde.md)                | Serde support and store export/import |
| [docs/design/benchmarks.md](docs/design/benchmarks.md)      | Benchmark suite and results           |
| [docs/design/testing.md](docs/design/testing.md)            | Test strategy                         |
| [docs/roadmap.md](docs/roadmap.md)                          | Development roadmap                   |

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

| Variable                       | Default | What it controls                                                    |
| ------------------------------ | ------- | ------------------------------------------------------------------- |
| `QJL_GPU_MIN_BATCH`            | 5000    | Float×sign `score()` GPU dispatch threshold. Set to `0` to force GPU. |
| `QJL_GPU_MIN_BATCH_COMPRESSED` | 100000  | Compressed `score_compressed()` GPU threshold. CPU is ~17 ns/vec.   |

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
