# turboquant

TurboQuant vector compression in Rust — sign-based hashing with
near-optimal distortion rate.

Compresses high-dimensional vectors (keys and values from attention
heads) into packed sign bits and quantized integers. Scores queries
against compressed stores via Hamming distance — no decompression,
no LLM, no GPU required.

Based on [TurboQuant](https://arxiv.org/abs/2504.19874) (Zandieh et al., 2025)
and [QJL](https://github.com/amirzandieh/QJL) (Zandieh et al., 2024).

## Status

Early development. See [docs/roadmap.md](docs/roadmap.md) for the
development plan.

## License

Licensed under either of

- Apache License, Version 2.0 ([LICENSE-APACHE](LICENSE-APACHE))
- MIT License ([LICENSE-MIT](LICENSE-MIT))

at your option.
