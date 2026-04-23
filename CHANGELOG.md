# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.2.0] — TBD

**Breaking change:** all public functions now return `Result<T, QjlError>`
instead of panicking on invalid input.

### Added

- `QjlError` enum with variants: `DimensionMismatch`, `InvalidSketchDim`,
  `InvalidBitWidth`, `NonFiniteInput`, `OutlierIndexOutOfRange`,
  `StoreMagicMismatch`, `StoreVersionMismatch`, `Io`
- `qjl_sketch::error::Result<T>` type alias
- `validate_finite` helper for NaN/infinity detection
- Input validation on all public API boundaries
- 18 new tests (7 error unit + 5 negative sketch/outlier + 6 negative integration)

### Changed

- `QJLSketch::new` returns `Result<Self>`
- `QJLSketch::quantize` returns `Result<CompressedKeys>`
- `QJLSketch::score` returns `Result<Vec<f32>>`
- `detect_outliers` returns `Result<Vec<u8>>`
- `quantize_values` returns `Result<CompressedValues>`
- `quantized_dot` returns `Result<f32>`
- `KeyQuantizer::new`, `build_sketch`, `update`, `attention_score` return `Result`
- `KeyStore` and `ValueStore` methods return `qjl_sketch::error::Result`
- Store config `read_from` returns `QjlError::StoreMagicMismatch` /
  `StoreVersionMismatch` instead of `io::Error`

## [0.1.0] — 2025-07-23

First release. CPU-only QJL compression, scoring, and persistence.

### Core Algorithms

- `QJLSketch` — random projection matrix with QR orthogonalization,
  deterministic from seed
- QJL sign-based key quantization with outlier/inlier separation
- Score computation via signed dot product with `sqrt(π/2)/s` scale
  factor — matches the QJL CUDA kernel exactly
- Outlier detection — top-k dimension norms across group
- Min-max value quantization (2-bit and 4-bit) with i32 bit-packing
- Fused dequantize + weighted dot product
- Streaming `KeyQuantizer` — batch and one-at-a-time compression

### Persistence

- `KeyStore` — append-only compressed key storage with mmap loading
- `ValueStore` — append-only compressed value storage with mmap loading
- Two independent stores per directory (keys.bin/idx + values.bin/idx)
- Sketch params in index header — no config.bin, matrix recomputed
  from seed
- Zero-copy `KeyPageView` / `ValuePageView` into mmap'd data
- Content-hash staleness detection (blake3)
- Compaction with atomic rename
- Crash recovery: truncated tail detection, index-ahead-of-store
  filtering

### Quality

- 75 tests (54 unit + 12 persistence integration + 9 quality)
- Distortion < 0.35 at sketch_dim = 2 × head_dim
- Ranking preservation: Kendall's tau > 0.70, top-10 recall ≥ 0.55
- Outlier separation reduces distortion ≥ 20% on spiky vectors
- Score survives persistence round-trip (bit-exact)

### Benchmarks

- Score: 18 µs/page (32 vectors, d=128, s=256)
- Key quantize: 38 µs/vector
- Cold start: 221 µs for 100 pages
- Page lookup: 5 ns (binary search + mmap slice)
