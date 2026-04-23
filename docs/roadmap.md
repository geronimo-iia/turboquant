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
| 5 | Error types + input validation — `QjlError` enum, all public API returns `Result` | 18 |

93 tests. Published on [crates.io](https://crates.io/crates/qjl-sketch).

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
