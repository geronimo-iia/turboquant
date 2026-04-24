# Benchmarks

Machine: Apple M3 Max, Rust 1.95.0, release build, criterion defaults.

## Running

```bash
cargo bench                                        # all CPU benchmarks
cargo bench --bench gpu_score --features gpu        # GPU benchmarks
./scripts/bench.sh --gpu --save                     # full run with report
```

## CPU Scoring (d=128, s=256, 32 vec/page)

| Operation | Time |
|-----------|------|
| Score 1 page (64 vec) | 24 us |
| Score 10 pages | 178 us |
| Score 100 pages | 1.8 ms |
| Score 1000 pages | 17.8 ms |

Linear scaling. ~18 us per page, ~0.56 us per vector.

## Compression

| Operation | Time |
|-----------|------|
| Key quantize 32 vec (d=128, s=256) | 1.23 ms |
| Key quantize 128 vec | 4.90 ms |
| Key quantize 512 vec | 19.7 ms |
| Value quantize 256 elem (4-bit) | 716 ns |
| Value quantize 4096 elem (4-bit) | 10.9 us |
| Sketch creation 64x128 | 210 us |
| Sketch creation 128x256 | 1.20 ms |
| Sketch creation 128x512 | 2.48 ms |

Key quantization: ~38 us/vector. Value quantization: ~2.7 ns/element.
Sketch creation is one-time cost on open.

## Store I/O

| Operation | Time |
|-----------|------|
| Cold start (100 pages) | 217 us |
| Append 1 page | 14.7 ms |
| Page lookup | 5 ns |

## GPU Store Scoring (`--features gpu`)

`score_all_pages` uses float x sign scoring. With GPU, all vectors
across all pages are batched into a single WGPU compute dispatch.
Without GPU, falls back to `sketch.score()` per page.

GPU dispatch threshold: `QJL_GPU_MIN_BATCH=5000` (total vectors).

### d=64, s=128, 32 vec/page

| Pages | Vectors | CPU per-page | GPU batched | Speedup |
|-------|---------|-------------|-------------|---------|
| 10 | 320 | 77 us | 77 us * | 1x |
| 100 | 3,200 | 765 us | 771 us * | 1x |
| 1000 | 32,000 | 7.5 ms | 3.1 ms | **2.4x** |
| 10000 | 320,000 | 76.9 ms | 11.0 ms | **7.0x** |

### d=128, s=256, 32 vec/page

| Pages | Vectors | CPU per-page | GPU batched | Speedup |
|-------|---------|-------------|-------------|---------|
| 10 | 320 | 218 us | 227 us * | 1x |
| 100 | 3,200 | 2.21 ms | 2.26 ms * | 1x |
| 1000 | 32,000 | 21.8 ms | 3.3 ms | **6.6x** |
| 10000 | 320,000 | 225.8 ms | 15.5 ms | **14.6x** |

\* Below GPU_MIN_BATCH (5K vectors), auto-dispatch uses CPU.
GPU forced via `QJL_GPU_MIN_BATCH=0` for comparison.

### Analysis

- GPU overhead: ~2.1 ms per dispatch (constant, independent of batch size)
- GPU per-vector cost: near zero (parallel compute)
- CPU per-vector cost: ~7.7 us (d=64) or ~22 us (d=128)
- Higher dimensions benefit more from GPU (more float multiplications per vector)
- Crossover: ~5K vectors for d=64, ~3K vectors for d=128
- At 10K pages: **7x speedup** (d=64) to **14.6x speedup** (d=128)

### When GPU helps

GPU is beneficial when:
1. Total vectors across all pages >= 5000 (default threshold)
2. Higher vector dimensions (d=128+) see larger speedups
3. The store has many pages (1000+)

GPU is NOT beneficial when:
- Few pages (< ~150 pages at 32 vec/page = 4800 vectors)
- The auto-dispatch correctly falls back to CPU in these cases
