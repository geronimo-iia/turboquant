# Contributing to qjl-sketch

## Prerequisites

- Rust 1.95.0 — pinned in `.tool-versions` and `rust-toolchain.toml`.
  Install via [asdf](https://asdf-vm.com/) or [rustup](https://rustup.rs/).
- `git`

## Build and Test

```bash
cargo build                                # debug build
cargo test                                 # all tests (default features)
cargo test --features serde                # include serde round-trip tests
cargo test --features gpu                  # include GPU tests (adapter tests are #[ignore])
cargo test --features gpu -- --ignored     # run GPU adapter tests (requires GPU)
cargo clippy -- -D warnings                # lint — must pass with zero warnings
cargo clippy --features gpu -- -D warnings # lint with GPU feature
cargo fmt -- --check                       # check formatting
cargo fmt                                  # auto-format
cargo audit                                # security audit
```

## Feature Flags

| Flag | What it enables |
|------|----------------|
| `serde` | `Serialize`/`Deserialize` on public structs, store export/import |
| `gpu` | WGPU compute shaders for GPU-accelerated scoring |

Default features are empty — both are opt-in.

## Environment Variables

| Variable | Default | What it controls |
|----------|---------|------------------|
| `QJL_GPU_MIN_BATCH` | 5000 | Float×sign `score()` GPU dispatch threshold |
| `QJL_GPU_MIN_BATCH_COMPRESSED` | 100000 | Compressed `score_compressed()` GPU threshold |

## Benchmarks

```bash
cargo bench                                       # all CPU benchmarks
cargo bench --bench gpu_score --features gpu       # GPU vs CPU
./scripts/bench.sh --gpu --save                    # full run with report
```

## Documentation

| Area           | Location                                           |
| -------------- | -------------------------------------------------- |
| Algorithms     | [docs/design/algorithms/](docs/design/algorithms/) |
| Persistence    | [docs/design/persistence.md](docs/design/persistence.md) |
| Store API      | [docs/design/store.md](docs/design/store.md)       |
| Serde          | [docs/design/serde.md](docs/design/serde.md)       |
| Benchmarks     | [docs/design/benchmarks.md](docs/design/benchmarks.md) |
| Testing        | [docs/design/testing.md](docs/design/testing.md)   |
| Roadmap        | [docs/roadmap.md](docs/roadmap.md)                 |
| Release guide  | [docs/release.md](docs/release.md)                 |

## Adding a Feature

1. Read the relevant design doc in `docs/design/`.
2. Implement in the correct module — see `src/lib.rs` for the module map.
3. Write tests in the appropriate `tests/` subdirectory:
   - `tests/quality/` — statistical validation
   - `tests/persistence/` — store round-trip, compaction, crash recovery
   - `tests/serde.rs` — serde round-trip (feature-gated)
   - Unit tests in `src/` modules via `#[cfg(test)]`
4. If adding a GPU feature, gate behind `#[cfg(feature = "gpu")]` and
   mark adapter-requiring tests with `#[ignore]`.
5. Run the full check:
   ```bash
   cargo fmt --check
   cargo clippy -- -D warnings
   cargo clippy --features gpu -- -D warnings
   cargo test
   cargo test --features serde
   cargo test --features gpu
   cargo doc --no-deps
   ```

## Examples

```bash
cargo run --example basic_qjl
cargo run --example compressed_scoring
cargo run --example mse_quantization
cargo run --example serde_roundtrip --features serde
cargo run --example store_export_import --features serde
```

## No LLM Dependency Rule

qjl-sketch compresses and scores vectors. It makes zero LLM calls and
has no opinion about how vectors are produced. Do not add any LLM
client, tokenizer, or model-loading crate as a dependency.

## Release Process

See [docs/release.md](docs/release.md).
