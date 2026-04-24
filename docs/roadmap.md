# Roadmap

## Goal

A standalone Rust crate (`qjl-sketch`) for vector compression and
scoring — QJL sign-based hashing, Lloyd-Max codebook quantization,
and TurboQuant MSE-optimal pipeline. CPU-first, GPU-optional.

## Completed

| Phase | What | Version |
|-------|------|---------|
| 0 | Project scaffold — CI, release, dependabot, docs, licenses | 0.1.0 |
| 1 | Core algorithms — sketch, outliers, quantize, score, values, streaming quantizer | 0.1.0 |
| 2 | Quality validation — distortion, ranking, outlier benefit, value accuracy | 0.1.0 |
| 3 | Persistence — KeyStore, ValueStore, staleness, compaction, crash recovery | 0.1.0 |
| 4 | Benchmarks — score, compress, store latency with criterion | 0.1.0 |
| 5 | Error types + input validation — `QjlError` enum, all public API returns `Result` | 0.2.0 |
| 6 | Lloyd-Max codebook — optimal scalar quantization (1–8 bit) | 0.3.0 |
| 7 | Math helpers — lgamma, beta_pdf, normal_icdf, Simpson's rule | 0.3.0 |
| 8 | Compressed-vs-compressed scoring — Hamming cosine estimator | 0.3.0 |
| 9 | Rust 2024 edition upgrade | 0.3.0 |
| 10 | MSE-optimal quantization — RandomRotation + Lloyd-Max per-coordinate | 0.4.0 |
| 11 | Serde support — feature-gated, streaming store export/import | 0.4.0 |
| 12 | Examples — basic_qjl, compressed_scoring, mse_quantization, serde_roundtrip, store_export_import | 0.4.0 |
| 13 | GPU acceleration — WGPU compute shader, runtime dispatch, store-level batch scoring | 0.5.0 |

162+ tests (with `--features serde,gpu`). Published on [crates.io](https://crates.io/crates/qjl-sketch).

## Next

### Full TurboQuant pipeline (`turbo.rs`)

Combine rotation + Lloyd-Max + QJL residual correction for the
complete TurboQuant two-stage pipeline (b-1 bits codebook + 1 bit
residual sign = b bits total per coordinate).

See [docs/prompts/two-stage-tasks.md](prompts/two-stage-tasks.md)
for the design and layering plan.

### Performance optimization

- [ ] SIMD: restructure `signed_dot` to process 8 bits per iteration
- [ ] Batch projection as GEMM via nalgebra BLAS
- [ ] `rayon` parallelism for multi-page scoring
- [ ] Batch append (amortize fsync cost)

### GPU — further optimization

- [ ] GPU-accelerated query projection (matrix-vector multiply)
- [ ] Buffer reuse / pre-allocation across queries
- [ ] Benchmark-calibrated `GPU_MIN_BATCH` per platform

### Other

- [ ] W_q / W_k / W_v weight loading from GGUF or safetensors

Pipeline integration (BM25 pre-filter + QJL rerank) lives in
the [llm-wiki](https://github.com/geronimo-iia/llm-wiki) project.
