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
- **Min-max value quantization** — 2-bit or 4-bit with i32 bit-packing
- **Score without decompressing** — signed dot product with
  `sqrt(π/2)/s` scale factor, matches the QJL CUDA kernel
- **Streaming quantizer** — batch or one-vector-at-a-time compression
- **Append-only persistence** — mmap-based KeyStore and ValueStore
  with zero-copy page views
- **Crash recovery** — truncated tail detection, index rebuild
- **Compaction** — reclaim dead space with atomic rename

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

Apple M-series, d=128, s=256, 32 vectors/page.

| Operation                          | Time   |
| ---------------------------------- | ------ |
| Score 1 page                       | 18 µs  |
| Score 100 pages                    | 1.8 ms |
| Score 1000 pages                   | 18 ms  |
| Key quantize (per vector)          | 38 µs  |
| Value quantize 4-bit (per element) | 2.4 ns |
| Cold start (100 pages)             | 221 µs |
| Page lookup                        | 5 ns   |

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

| Document                                                 | What it covers                |
| -------------------------------------------------------- | ----------------------------- |
| [docs/study.md](docs/study.md)                           | TurboQuant algorithm overview |
| [docs/design/algorithms.md](docs/design/algorithms.md)   | 7 algorithms with pseudocode  |
| [docs/design/persistence.md](docs/design/persistence.md) | Two-store file format         |
| [docs/design/store.md](docs/design/store.md)             | Store API and lifecycle       |
| [docs/design/benchmarks.md](docs/design/benchmarks.md)   | Benchmark suite and results   |
| [docs/design/testing.md](docs/design/testing.md)         | Test strategy                 |
| [docs/roadmap.md](docs/roadmap.md)                       | Development roadmap           |

## Building

```bash
cargo build                      # debug
cargo test                       # 93 tests (~8s)
cargo bench                      # criterion benchmarks (release)
cargo clippy -- -D warnings      # lint
```

Requires Rust 1.95+.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
