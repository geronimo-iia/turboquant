# Roadmap

## Goal

A standalone Rust crate (`turboquant`) that compresses vectors via the
TurboQuant pipeline and scores queries against compressed stores —
no LLM, no GPU, CPU-only. Testable end-to-end: compress vectors,
persist, reload, score, verify ranking preservation.

## Phase 0 — Project scaffold

Set up the crate, CI, and test harness.

- [x] `cargo init --lib turboquant` in `projects/turboquant/`
- [x] Cargo.toml: edition 2021, rust-version 1.95, MIT OR Apache-2.0
- [x] Dependencies: `nalgebra`, `rand`, `rand_distr`, `bytemuck`, `rayon`, `memmap2`, `blake3`
- [x] Dev-dependencies: `tempfile`, `approx`
- [x] `src/lib.rs` with empty module declarations
- [x] `.github/workflows/ci.yml` — cargo fmt, clippy, test, audit
- [x] `.github/workflows/release.yml` — test + publish to crates.io
- [x] `.github/workflows/dependabot.yml` — auto-merge patch/minor
- [x] `.github/dependabot.yml` — cargo + github-actions weekly
- [x] `.github/ISSUE_TEMPLATE/` — bug report, feature request, config
- [x] `.github/pull_request_template.md`
- [x] `rust-toolchain.toml` — pinned to 1.95
- [x] `rustfmt.toml`, `clippy.toml`, `audit.toml`
- [x] `.tool-versions` — rust 1.95.0
- [x] `LICENSE-MIT`, `LICENSE-APACHE`
- [x] `CHANGELOG.md`, `CONTRIBUTING.md`, `SECURITY.md`, `README.md`
- [x] `docs/` — README, roadmap, release guide
- [x] `docs/study/` — article, overview
- [x] `docs/design/` — algorithms, pipeline, persistence, testing
- [x] `tests/` directory structure matching `design/testing.md`
- [x] `cargo build` — clean
- [x] `cargo test` — passes (zero tests)


### Milestone: `cargo test` passes (zero tests, clean build)

## Phase 1 — Core algorithms (CPU, f32)

Implement each algorithm from `design/algorithms.md` with unit tests.
One module per concern. No persistence, no pipeline — pure math.

### 1a — Random projection

- [ ] `src/sketch.rs`: `QJLSketch` struct
- [ ] `QJLSketch::new(head_dim, sketch_dim, seed)` — Gaussian init
- [ ] QR orthogonalization per chunk (Algorithm 1)
- [ ] `proj_dir_score` and `proj_dir_quant` stored as `Vec<f32>`
- [ ] Tests: dimensions, orthogonality, determinism, different seeds

### 1b — Outlier detection

- [ ] `src/outliers.rs`: `detect_outliers(keys, count) → Vec<u8>`
- [ ] L2 norm per dimension across group, top-k selection (Algorithm 4)
- [ ] Tests: known outlier picked, count respected

### 1c — QJL quantization

- [ ] `src/quantize.rs`: `CompressedKeys` struct
- [ ] `QJLSketch::quantize(keys, outlier_indices) → CompressedKeys`
- [ ] Sign extraction, bit-packing 8 signs per u8 (Algorithm 2)
- [ ] Outlier/inlier separation, outlier norms
- [ ] Tests: output shape, bit-packing correctness, outlier separation, norms

### 1d — Score computation

- [ ] `src/score.rs`: `QJLSketch::score(query, compressed) → Vec<f32>`
- [ ] Query sketch projection
- [ ] Hamming distance via XOR + `u8::count_ones()` (Algorithm 3)
- [ ] Norm-weighted cosine estimate
- [ ] Tests: identical vectors, orthogonal vectors, sign preserved, popcount

### 1e — Value quantization

- [ ] `src/values.rs`: `CompressedValues` struct
- [ ] `quantize_values(values, group_size, bits) → CompressedValues`
- [ ] Min-max scalar quantization + i32 bit-packing (Algorithm 5)
- [ ] `quantized_matmul(weights, compressed) → Vec<f32>` (Algorithm 6)
- [ ] Tests: round-trip error bound, 4-bit/2-bit range, matmul accuracy

### 1f — Streaming quantizer

- [ ] `src/quantizer.rs`: `KeyQuantizer` struct
- [ ] `build_sketch(keys)` — batch compress (Algorithm 7 init)
- [ ] `update(key)` — append one vector, flush on buffer full (Algorithm 7)
- [ ] `attention_score(query) → Vec<f32>` — score against full state
- [ ] Tests: stream matches batch, residual buffer, buffer flush

### Milestone: `cargo test unit` — all algorithms correct in isolation

## Phase 2 — Quality validation

Statistical tests proving our implementation preserves the properties
that TurboQuant guarantees. Each test is self-contained (no external
fixtures). Quality tests are `#[ignore]` by default — they run 10K+
iterations.

### 2a — Rotation preserves geometry

The orthogonal projection must not distort vectors.

- [ ] `test_rotation_preserves_norm` — for 1K random vectors,
      assert `||proj @ v|| / ||v||` is within [0.95, 1.05] of `sqrt(d)`
      (the scale factor)
- [ ] `test_rotation_preserves_inner_product` — for 1K random (q, k)
      pairs, compute `dot(proj@q, proj@k)` vs `d * dot(q, k)`.
      Assert relative error < 0.1 on average.

### 2b — Sign quantization distortion

The paper claims ~2.7x optimal distortion rate.

- [ ] `test_distortion_rate` — for 10K random (q, k) pairs:
      `distortion = E[|dot(q,k) - score(q, compress(k))|²] / E[|dot(q,k)|²]`
      Assert distortion < 0.20 at sketch_dim = 2 * head_dim.
- [ ] `test_distortion_decreases_with_sketch_dim` — measure distortion
      at sketch_dim = d, 2d, 4d. Assert monotonically decreasing.

### 2c — Ranking preservation

The practical test: does compression preserve which keys are most
relevant?

- [ ] `test_top_k_recall` — 1 query + 200 keys. Compute exact top-10
      and compressed top-10. Assert recall ≥ 0.7 (at least 7 of 10
      match). Run 100 trials, assert mean recall ≥ 0.8.
- [ ] `test_kendall_tau` — 1 query + 100 keys. Compute Kendall's tau
      between exact ranking and compressed ranking. Assert tau > 0.85
      averaged over 50 trials.

### 2d — Value quantization accuracy

- [ ] `test_value_quantized_matmul_error` — for 1K random
      (weights, values) pairs, compare exact `weights @ values` vs
      `quantized_dot(weights, compress(values))`. Assert mean relative
      error < 0.05 at 4-bit, < 0.15 at 2-bit.

### 2e — Outlier separation benefit

Outlier separation should improve score quality on vectors with spikes.

- [ ] `test_outlier_vs_no_outlier` — generate vectors with 2 outlier
      dimensions (10x magnitude). Compare distortion with
      outlier_count=2 vs outlier_count=0. Assert outlier separation
      reduces distortion by at least 20%.

### Milestone: `cargo test quality` — ranking preservation proven

## Phase 3 — Persistence

Implement the packed-file store from `design/persistence.md`.

### 3a — Config file

- [ ] `src/store/config.rs`: `StoreConfig` struct
- [ ] Write/read `config.bin` — header + projection matrices
- [ ] mmap-based loading via `memmap2`

### 3b — Packed store

- [ ] `src/store/store.rs`: `KVStore` struct
- [ ] `append(slug_hash, content_hash, compressed, generation)`
- [ ] `store.bin` append-only write with entry header
- [ ] `store.idx` — sorted index, binary search lookup
- [ ] `get_page(slug_hash) → PageView` — zero-copy slice into mmap
- [ ] Tests: write-read round-trip, score survives persistence

### 3c — Update and staleness

- [ ] `is_fresh(slug_hash, current_content_hash) → bool`
- [ ] Append with higher generation, old entry becomes dead
- [ ] `dead_bytes()` / `live_bytes()` tracking
- [ ] Tests: update overwrites old, dead space tracked, staleness detection

### 3d — Compaction

- [ ] `compact()` — rewrite live entries, update index
- [ ] Atomic rename for crash safety
- [ ] Tests: reclaims space, preserves scores, all pages still readable

### 3e — Crash recovery

- [ ] Detect truncated tail on open (magic check), truncate
- [ ] Rebuild index from store if index is stale/missing
- [ ] Tests: truncated tail, index ahead of store

### Milestone: `cargo test persistence` — compress, persist, reload, score = same result

## Phase 4 — Pipeline

Wire everything into the query pipeline from `design/pipeline.md`.

- [ ] `src/pipeline.rs`: `Pipeline` struct
- [ ] `Pipeline::compress_page(tokens, slug) → ()` — project + quantize + store
- [ ] `Pipeline::query(query_tokens, top_k) → Vec<PageScore>` — project
      query, scan store, rank by attention score
- [ ] `Pipeline::recompress(slug)` — re-compress a single page
- [ ] `Pipeline::rebuild()` — re-compress all pages from scratch
- [ ] Tests: smoke test, relevant page ranks high, incremental update,
      empty store, single token page

### Milestone: `cargo test e2e` — search query → ranked pages with scores

## Phase 5 — Performance

Not needed for correctness, but needed for practical use.

- [ ] SIMD popcount: `std::arch` for `_popcnt64` on x86, fallback
      to `u8::count_ones()`
- [ ] Batch projection as GEMM via `nalgebra` BLAS
- [ ] `rayon` parallelism for multi-head score computation
- [ ] Benchmark suite: `benches/` with `criterion`
      — score latency vs. page count (100, 1K, 10K)
      — compress throughput (pages/sec)
      — cold start time (mmap open + first query)

### Milestone: benchmark numbers documented, no regressions in CI

## Future

- GPU score kernel via `wgpu` compute shaders
- Integration with llm-wiki: `wiki_ingest` triggers compress,
  `wiki_search` optionally uses TurboQuant scores as reranker
- W_q / W_k / W_v weight loading from GGUF or safetensors
- Hybrid mode: BM25 pre-filter + TurboQuant rerank

## Project Structure (target)

```
projects/turboquant/
├── study/                  ← source material
│   ├── README.md
│   └── article.md
├── design/                 ← architecture decisions
│   ├── algorithms.md
│   ├── persistence.md
│   ├── pipeline.md
│   └── testing.md
├── src/
│   ├── lib.rs
│   ├── sketch.rs           ← QJLSketch (projection, quantize, score)
│   ├── outliers.rs          ← outlier detection
│   ├── quantize.rs          ← CompressedKeys, sign hashing
│   ├── score.rs             ← score computation (Hamming + norms)
│   ├── values.rs            ← CompressedValues, quantized matmul
│   ├── quantizer.rs         ← KeyQuantizer (batch + streaming)
│   ├── store/
│   │   ├── mod.rs
│   │   ├── config.rs        ← StoreConfig, config.bin
│   │   └── store.rs         ← KVStore, append, compact, mmap
│   └── pipeline.rs          ← Pipeline (compress, query, recompress)
├── tests/
│   ├── unit/
│   ├── quality/
│   ├── persistence/
│   └── e2e/
├── benches/
│   └── score.rs
├── article.md
├── Cargo.toml
├── LICENSE-APACHE
├── LICENSE-MIT
└── README.md
```
