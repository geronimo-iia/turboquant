# Release Process

## Prerequisites

- All tests pass (`cargo test && cargo test -- --ignored`)
- `cargo clippy -- -D warnings` clean
- `cargo fmt -- --check` clean
- `cargo audit` clean
- CHANGELOG.md updated with release date and notes

## Steps

1. Update version in `Cargo.toml`
2. Update `[Unreleased]` section in `CHANGELOG.md` → `[x.y.z] — YYYY-MM-DD`
3. Commit: `git commit -am "release: vx.y.z"`
4. Tag: `git tag vx.y.z`
5. Push: `git push origin main --tags`

The `Release` GitHub Action will:
- Run full test suite (including quality tests)
- Publish to crates.io

## Versioning

Follows [Semantic Versioning](https://semver.org/spec/v2.0.0.html):

- **0.x.y** — pre-1.0, breaking changes bump minor
- **1.0.0** — stable API: compress, score, persist, query
