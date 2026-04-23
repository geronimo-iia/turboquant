# Contributing to turboquant

## Prerequisites

- Rust 1.95.0 — pinned in `.tool-versions` and `rust-toolchain.toml`.
  Install via [asdf](https://asdf-vm.com/) or [rustup](https://rustup.rs/).
- `git`

## Build and Test

```bash
cargo build                      # debug build
cargo test                       # all tests (unit + quality, ~7s)
cargo clippy -- -D warnings      # lint — must pass with zero warnings
cargo fmt -- --check             # check formatting
cargo fmt                        # auto-format
cargo audit                      # security audit
```

## Documentation

| Area           | Location                                |
| -------------- | --------------------------------------- |
| Study docs     | [docs/study/](docs/study/README.md)     |
| Design docs    | [docs/design/](docs/design/)             |
| Algorithms     | [docs/design/algorithms.md](docs/design/algorithms.md) |
| Roadmap        | [docs/roadmap.md](docs/roadmap.md)       |
| Release guide  | [docs/release.md](docs/release.md)      |

## Adding a Feature

1. Read the relevant design doc in `docs/design/`.
2. Implement in the correct module — see `src/lib.rs` for the module map.
3. Write tests in the appropriate `tests/` subdirectory:
   - `tests/unit/` — algorithm correctness
   - `tests/quality/` — statistical validation (`#[ignore]` by default)
   - `tests/persistence/` — store round-trip, compaction, crash recovery
   - `tests/e2e/` — full pipeline
4. Run `cargo test`, `cargo clippy -- -D warnings`, `cargo fmt -- --check`.

## No LLM Dependency Rule

turboquant compresses and scores vectors. It makes zero LLM calls and
has no opinion about how vectors are produced. Do not add any LLM
client, tokenizer, or model-loading crate as a dependency.

## Release Process

See [docs/release.md](docs/release.md).
