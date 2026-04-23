# Roadmap

## Goal

A standalone Rust crate (`qjl-sketch`) that compresses vectors via
QJL sign-based hashing and scores queries against compressed stores —
no LLM, no GPU, CPU-only.

## Completed

| Phase | What | Tests |
|-------|------|-------|
| 0 | Project scaffold — CI, release, dependabot, docs, licenses | — |
| 1 | Core algorithms — sketch, outliers, quantize, score, values, streaming quantizer | 32 |
| 2 | Quality validation — distortion, ranking, outlier benefit, value accuracy | 9 |
| 3 | Persistence — KeyStore, ValueStore, staleness, compaction, crash recovery | 34 |
| 4 | Benchmarks — score, compress, store latency with criterion | — |

75 tests. Published on [crates.io](https://crates.io/crates/qjl-sketch).

## Active

### Error types + Input validation

Replace panics with a proper error enum and validate all public API inputs.
Done together since validation returns the new error type.

**Step 1 — Error enum**

- [ ] `src/error.rs`: define `QjlError` enum
- [ ] Variants: `DimensionMismatch { expected, got }`,
      `InvalidSketchDim(usize)`, `InvalidBitWidth(u8)`,
      `NonFiniteInput { context }`, `StoreMagicMismatch`,
      `StoreVersionMismatch { expected, got }`, `Io(std::io::Error)`
- [ ] `pub type Result<T> = std::result::Result<T, QjlError>`
- [ ] `impl Display`, `impl Error`, `impl From<std::io::Error>`
- [ ] Add `pub mod error` to `lib.rs`
- [ ] Tests: Display output, From<io::Error> conversion

**Step 2 — Migrate sketch.rs**

- [ ] `QJLSketch::new` returns `Result<Self>`
      — validate head_dim > 0, sketch_dim > 0, divisible by 8,
        outlier_sketch_dim <= sketch_dim
- [ ] `matvec`, `l2_norm` stay infallible (internal, trusted inputs)
- [ ] Update sketch unit tests

**Step 3 — Migrate outliers.rs**

- [ ] `detect_outliers` returns `Result<Vec<u8>>`
      — validate keys.len() == group_size * head_dim,
        count <= head_dim, head_dim <= 256
- [ ] Update outlier unit tests

**Step 4 — Migrate quantize.rs**

- [ ] `QJLSketch::quantize` returns `Result<CompressedKeys>`
      — validate keys.len() == num_vectors * head_dim,
        all values finite, all outlier indices < head_dim
- [ ] Update quantize unit tests

**Step 5 — Migrate score.rs**

- [ ] `QJLSketch::score` returns `Result<Vec<f32>>`
      — validate query.len() == head_dim, all values finite
- [ ] Update score unit tests

**Step 6 — Migrate values.rs**

- [ ] `quantize_values` returns `Result<CompressedValues>`
      — validate values.len() divisible by group_size,
        bits == 2 or 4, all values finite
- [ ] `quantized_dot` returns `Result<f32>`
      — validate weights.len() == num_elements, all finite
- [ ] Update values unit tests

**Step 7 — Migrate quantizer.rs**

- [ ] `KeyQuantizer::new` returns `Result<Self>`
      — validate buffer_size divisible by group_size
- [ ] `build_sketch`, `update`, `attention_score` return `Result`
- [ ] Update quantizer unit tests

**Step 8 — Migrate store**

- [ ] `store/config.rs`: replace `io::Error::new(InvalidData, ...)`
      with `QjlError::StoreMagicMismatch` / `StoreVersionMismatch`
- [ ] `KeyStore::create`, `open`, `append` return `Result`
      — validate compressed data lengths match config
- [ ] `ValueStore::create`, `open`, `append` return `Result`
      — same validation
- [ ] Update store unit tests

**Step 9 — Migrate integration tests**

- [ ] Update `tests/quality/*.rs` — add `.unwrap()` on Result returns
- [ ] Update `tests/persistence/main.rs` — same
- [ ] Add negative tests: dimension mismatch, invalid bit width,
      NaN input, zero dimension

**Step 10 — Verify**

- [ ] `cargo test` — all 75+ tests pass
- [ ] `cargo clippy -- -D warnings` — clean
- [ ] No remaining `assert!` on user input in public functions

**Step 11 — benchmark

- [ ] update benchmark code

**Step 12 — Documentation + release**

- [ ] Update `docs/design/algorithms.md` — all function signatures show `Result`
- [ ] Update `docs/design/store.md` — error handling section reflects `QjlError`
- [ ] Update `docs/design/testing.md` — add negative test descriptions
- [ ] Update `README.md` — usage examples with `?` / `.unwrap()`
- [ ] Update `CHANGELOG.md` — breaking change: public API returns `Result`
- [ ] Bump version to 0.2.0 in `Cargo.toml`

## Future

### Performance optimization

- [ ] SIMD: restructure `signed_dot` to process 8 bits per iteration
- [ ] Batch projection as GEMM via nalgebra BLAS
- [ ] `rayon` parallelism for multi-page scoring
- [ ] Batch append (amortize fsync cost)

### Serde support

Add `Serialize`/`Deserialize` on public structs for debug dumps and interop.

- [ ] Add `serde = { version = "1", features = ["derive"] }` to dependencies
- [ ] Derive `Serialize, Deserialize` on `CompressedKeys`
- [ ] Derive `Serialize, Deserialize` on `CompressedValues`
- [ ] Derive `Serialize, Deserialize` on `KeysConfig`, `ValuesConfig`
- [ ] Derive `Serialize, Deserialize` on `IndexEntry`, `IndexMeta`
- [ ] `QJLSketch`: serialize as params only (head_dim, sketch_dim,
      outlier_sketch_dim, seed) — not the matrices
- [ ] Tests: serde round-trip for each struct (serialize → deserialize → equal)
- [ ] Feature-gate behind `serde` feature flag to keep default deps minimal

### Other

- GPU score kernel via `wgpu` compute shaders
- W_q / W_k / W_v weight loading from GGUF or safetensors

Pipeline integration (BM25 pre-filter + QJL rerank) lives in
the [llm-wiki](https://github.com/geronimo-iia/llm-wiki) project.
