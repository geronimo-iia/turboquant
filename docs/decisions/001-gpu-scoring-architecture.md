# ADR-001: GPU Scoring Architecture

Date: 2025-07-24
Status: Accepted

## Context

We added WGPU GPU acceleration for vector scoring. Initial
implementation dispatched to GPU from `score()` and `score_compressed()`
per-call, and from `score_all_pages()` per-page. Benchmarks on
Apple M3 Max revealed this approach was counterproductive.

## Benchmark Data (Apple M3 Max)

### GPU overhead

Single GPU dispatch (buffer upload + kernel launch + readback): **~2.1 ms**

### Per-call scoring — GPU never wins

| Method | CPU per-vector | GPU overhead | Breakeven |
|--------|---------------|-------------|-----------|
| `score()` float×sign | 0.33 µs | 2.1 ms | ~7K vectors |
| `score_compressed()` byte XOR | 7 ns | 2.1 ms | ~300K vectors |

A single page has 32 vectors. GPU overhead (2.1 ms) vs CPU cost
(32 × 0.33 µs = 10.6 µs for float×sign, 32 × 7 ns = 224 ns for
compressed). GPU is 200-9000x slower per page.

### Per-page store scoring — GPU makes it worse

| Pages | CPU compressed | GPU per-page float×sign |
|-------|---------------|------------------------|
| 10 | 18 µs | 20.5 ms (1000x slower) |
| 100 | 97 µs | 203 ms (2000x slower) |
| 1000 | 7.8 ms | 2.16 s (277x slower) |

Root cause: 1000 pages × 2.1 ms GPU overhead per dispatch = 2.1 s.

### Batched store scoring — GPU can win

By collecting ALL vectors across ALL pages into contiguous buffers
and dispatching ONE GPU kernel:

- 1000 pages × 32 vectors = 32K vectors
- GPU: 2.1 ms overhead + ~0.5 ms compute ≈ **2.6 ms** (estimated)
- CPU compressed: 7.8 ms

Single-dispatch batch should be **~3x faster** than CPU for 1000+ pages.

## Decision

### GPU only in `score_all_pages` batch path

1. **`score()` — always CPU.** Per-page calls (32 vectors) can never
   amortize 2.1 ms GPU overhead. Removed GPU dispatch.

2. **`score_compressed()` — always CPU.** 7 ns/vector on CPU. GPU
   can never beat this. Removed GPU dispatch.

3. **`score_compressed_pair()` — always CPU.** Single pair, no GPU path.

4. **`score_all_pages()` — always float x sign.**
   With GPU: batches all sign bits + norms across all pages into
   contiguous buffers, single GPU dispatch, 5.6x faster.
   Without GPU: `sketch.score()` per page on CPU.
   Both paths produce identical float x sign scores.
   GPU activates when total vectors >= `QJL_GPU_MIN_BATCH` (5000).

### Single environment variable

- `QJL_GPU_MIN_BATCH` (default 5000) — controls `score_all_pages`
  GPU dispatch threshold. Only env var needed since GPU only
  activates in one place.
- Removed `QJL_GPU_MIN_BATCH_COMPRESSED` — no longer used.

### GPU kernel

Only `score_float_sign.wgsl` is kept. The compressed Hamming kernel
(`score.wgsl`) was removed -- CPU byte XOR + popcount at 7 ns/vector
is unbeatable on Apple Silicon.

## Consequences

- Per-page scoring is always fast (CPU, no GPU overhead)
- Store-level scoring benefits from GPU when enough pages are present
- Simple mental model: "GPU = store-level batch only"
- One env var to tune, not two
- GPU feature flag still opt-in, zero overhead when disabled

## Alternatives Considered

1. **GPU dispatch in `score()` with high threshold** — rejected because
   the threshold (5K+) is never reached by per-page calls, making
   the dispatch check dead code.

2. **Remove GPU entirely** — rejected because batched `score_all_pages`
   is a valid use case where GPU can win.

3. **Separate thresholds per method** — rejected as unnecessary
   complexity. Only one method uses GPU.
