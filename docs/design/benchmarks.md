# Benchmarks

## Running

Benchmarks use [criterion](https://bheisler.github.io/criterion.rs/book/)
and run in release mode automatically.

```bash
cargo bench                                # all benchmarks (CPU)
cargo bench --bench score                  # score latency only
cargo bench --bench compress               # compression throughput only
cargo bench --bench store                  # store I/O only
cargo bench --bench gpu_score --features gpu  # GPU vs CPU scoring
```

Or use the runner script to collect all results into a report:

```bash
./scripts/bench.sh              # CPU only
./scripts/bench.sh --gpu        # include GPU benchmarks
./scripts/bench.sh --gpu --save # save report to artifacts/bench-<timestamp>/
```

HTML reports are generated in `target/criterion/`. Open
`target/criterion/report/index.html` for interactive charts.

## What's Measured

### Score latency (`benches/score.rs`)

How long it takes to score a query against compressed pages.

| Benchmark | What it measures |
|-----------|------------------|
| `score_latency/pages/N` | Score 1 query against N pages (32 vectors each) |
| `score_single_page_64vec` | Score 1 query against 1 page (64 vectors) |

### Compression throughput (`benches/compress.rs`)

How long it takes to compress vectors.

| Benchmark | What it measures |
|-----------|------------------|
| `key_quantize/vectors/N` | QJL quantize N vectors (d=128, s=256) |
| `value_quantize/elements_4bit/N` | Min-max 4-bit quantize N elements |
| `sketch_creation/dim/DxS` | QJLSketch::new (QR decomposition) |

### Store I/O (`benches/store.rs`)

Persistence performance.

| Benchmark | What it measures |
|-----------|------------------|
| `cold_start_100_pages` | Open KeyStore with 100 pages (mmap + index load) |
| `append_single_page` | Append one page (serialize + write + fsync + index rewrite) |
| `get_page_from_100` | Binary search + mmap slice for one page |

### GPU scoring (`benches/gpu_score.rs`, requires `--features gpu`)

CPU vs GPU scoring comparison.

| Benchmark | What it measures |
|-----------|------------------|
| `score_compressed_cpu_vs_gpu/cpu/N` | Compressed scoring of N vector pairs (dispatches CPU or GPU based on threshold) |
| `score_all_pages/pages/N` | `KeyStore::score_all_pages` for N pages (32 vectors each) |

Use `QJL_GPU_MIN_BATCH=0` to force GPU path at all batch sizes.

## Baseline Results

Machine: Apple M-series, release build, criterion defaults.
Parameters: d=128, s=256, os=64, 32 vectors/page.

Measured on v0.1.0. The v0.2.0 `Result` migration adds no measurable
overhead (validation checks are O(n) scans on inputs that are already
being iterated).

These are Phase 4 optimization targets — deferred until llm-wiki
integration reveals real-world bottlenecks.

### Score

| Benchmark | Time |
|-----------|------|
| score 1 page (64 vec) | 24 µs |
| score 10 pages | 181 µs |
| score 100 pages | 1.8 ms |
| score 1000 pages | 18 ms |

Linear scaling. ~18 µs per page (32 vectors). The hot loop is
`signed_dot`: iterate sketch_dim projections, extract sign bit,
multiply by query sketch float, accumulate.

### Compression

| Benchmark | Time |
|-----------|------|
| key quantize 32 vec | 1.23 ms |
| key quantize 128 vec | 4.94 ms |
| key quantize 512 vec | 19.7 ms |
| value quantize 256 elem (4-bit) | 756 ns |
| value quantize 4096 elem (4-bit) | 10.9 µs |
| sketch creation 64×128 | 210 µs |
| sketch creation 128×256 | 1.21 ms |
| sketch creation 128×512 | 2.47 ms |

Key quantization is ~38 µs/vector (dominated by the d×s projection).
Value quantization is ~2.7 ns/element. Sketch creation is one-time
cost on open.

### Store I/O

| Benchmark | Time |
|-----------|------|
| cold start (100 pages) | 220 µs |
| append 1 page | 16 ms |
| get_page lookup | 5 ns |

Cold start is fast (mmap + index read). Append is slow due to fsync +
index rewrite on every call — batching would help. Page lookup is
essentially free (binary search + pointer arithmetic).

## Optimization Opportunities

Based on the numbers:

1. **`signed_dot` is the score bottleneck.** Currently iterates bit-by-bit.
   Processing 8 bits at a time (unpack byte, multiply 8 floats) would
   reduce loop iterations 8x.

2. **Key quantization is projection-dominated.** The inner loop is
   `d × s` multiply-accumulate per vector. BLAS GEMM would help for
   large batches.

3. **Append fsync is expensive.** Batching multiple appends before
   fsync would amortize the cost. Not a scoring bottleneck.

4. **`u8::count_ones()` already compiles to `popcnt` on x86.** SIMD
   popcount may not help unless we restructure to process 64 bits at
   a time.

5. **GPU dispatch overhead.** On Apple M3 Max, GPU overhead is ~1.7 ms
   (buffer upload + kernel launch + readback). Per-vector GPU cost is
   near zero above 10K vectors. Breakeven vs CPU compressed scoring
   is ~3K-5K vectors. The `QJL_GPU_MIN_BATCH` threshold defaults to
   5000. Run `cargo bench --bench gpu_score --features gpu` to
   calibrate for your hardware.

### GPU Scoring (Apple M3 Max)

| Benchmark | Time |
|-----------|------|
| score_compressed 100 vec | 1.72 ms |
| score_compressed 1K vec | 1.85 ms |
| score_compressed 10K vec | 2.28 ms |
| score_compressed 100K vec | 3.44 ms |
| score_all_pages 10 pages | 17.7 µs |
| score_all_pages 100 pages | 91.2 µs |
| score_all_pages 1000 pages | 840 µs |

