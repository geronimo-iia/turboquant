# Documentation

## Study

Source material and research analysis.

| Document                           | What it covers                      |
| ---------------------------------- | ----------------------------------- |
| [api-overview.md](api-overview.md) | Full public API by use case         |
| [study.md](study.md)               | TurboQuant overview, source article |

## Design

Architecture and implementation decisions.
See [design/README.md](design/README.md) for the full index.

| Document                                          | What it covers                                                 |
| ------------------------------------------------- | -------------------------------------------------------------- |
| [design/algorithms/](design/algorithms/README.md) | Algorithm catalog: projection, quantization, scoring, codebook |
| [design/persistence.md](design/persistence.md)    | Two-store file format, mmap loading, compaction                |
| [design/store.md](design/store.md)                | Store API usage, lifecycle, error handling, crash safety       |
| [design/serde.md](design/serde.md)                | Serde support, feature flag, store export/import               |
| [design/testing.md](design/testing.md)            | Test strategy, 167 tests across 5 categories                   |
| [benchmarks.md](benchmarks.md)                    | Benchmark suite, baseline results, GPU analysis                |

## Decisions

Architecture Decision Records (ADRs).

| ADR                                              | Decision                                                 |
| ------------------------------------------------ | -------------------------------------------------------- |
| [001](decisions/001-gpu-scoring-architecture.md) | GPU only in `score_all_pages` batch path, never per-page |

## Guides

- [roadmap.md](roadmap.md) — Phased development plan
- [release.md](release.md) — How to cut a release
