# Benchmarks

## Running

Benchmarks use [criterion](https://bheisler.github.io/criterion.rs/book/)
and run in release mode automatically.

```bash
cargo bench                      # all benchmarks
cargo bench --bench score        # score latency only
cargo bench --bench compress     # compression throughput only
cargo bench --bench store        # store I/O only
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

## Baseline Results

Machine: Apple M-series, release build, criterion defaults.
Parameters: d=128, s=256, os=64, 32 vectors/page.

### Score

| Benchmark | Time |
|-----------|------|
| score 1 page (64 vec) | 25 µs |
| score 10 pages | 182 µs |
| score 100 pages | 1.8 ms |
| score 1000 pages | 18 ms |

Linear scaling. ~18 µs per page (32 vectors). The hot loop is
`signed_dot`: iterate sketch_dim projections, extract sign bit,
multiply by query sketch float, accumulate.

### Compression

| Benchmark | Time |
|-----------|------|
| key quantize 32 vec | 1.2 ms |
| key quantize 128 vec | 4.9 ms |
| key quantize 512 vec | 19.7 ms |
| value quantize 256 elem (4-bit) | 670 ns |
| value quantize 4096 elem (4-bit) | 9.8 µs |
| sketch creation 128×256 | 1.2 ms |
| sketch creation 128×512 | 2.5 ms |

Key quantization is ~38 µs/vector (dominated by the d×s projection).
Value quantization is ~2.4 ns/element. Sketch creation is one-time
cost on open.

### Store I/O

| Benchmark | Time |
|-----------|------|
| cold start (100 pages) | 221 µs |
| append 1 page | 14.4 ms |
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

