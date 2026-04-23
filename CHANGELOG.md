# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

## [0.1.0] — TBD

First release. CPU-only TurboQuant compression and scoring.

### Added

- `QJLSketch` — random projection matrix with QR orthogonalization
- `CompressedKeys` — sign-based key quantization with outlier separation
- `CompressedValues` — min-max scalar quantization with bit-packing
- `KeyQuantizer` — batch and streaming key compression
- Score computation via Hamming distance (XOR + popcount)
- Outlier detection (top-k norms per dimension)
- `KVStore` — append-only packed file persistence with mmap loading
- Staleness detection via content hashing (blake3)
- Compaction with atomic rename
- `Pipeline` — compress pages, query with ranked results
