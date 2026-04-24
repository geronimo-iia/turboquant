# Design Documents

Architecture and implementation decisions for qjl-sketch.

| Document | What it covers |
|----------|----------------|
| [algorithms/](algorithms/README.md) | Algorithm catalog (11 algorithms) |
| [persistence.md](persistence.md) | Two-store file format, mmap loading, compaction |
| [store.md](store.md) | Store API usage, lifecycle, error handling, crash safety |
| [serde.md](serde.md) | Serde support, feature flag, store export/import |
| [testing.md](testing.md) | Test strategy, 167 tests across 5 categories |
