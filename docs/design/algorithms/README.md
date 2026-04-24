# Algorithms

Reference: `amirzandieh/QJL` (Python/CUDA), paper arXiv:2504.19874.
Verified against `qjl_score_kernel.cu` and `qjl_quant_kernel.cu`.

## Architecture

```
src/
├── sketch.rs       QJLSketch — projection matrix, quantize, score
├── outliers.rs     detect_outliers — top-k norms per dimension
├── quantize.rs     CompressedKeys — sign hashing, bit-packing
├── score.rs        score, score_compressed, hamming_similarity
├── values.rs       CompressedValues — min-max quantization, fused matmul
├── quantizer.rs    KeyQuantizer — batch + streaming wrapper
├── math.rs         Numerical helpers — lgamma, beta_pdf, normal_icdf, Simpson's rule
└── codebook.rs     Codebook — Lloyd-Max optimal scalar quantization
```

## Algorithm index

| # | Name | Source file | Doc |
|---|------|-------------|-----|
| 1 | [Random Projection Matrix](01-random-projection.md) | `sketch.rs` | QR-orthogonalized Gaussian projection |
| 2 | [QJL Quantization](02-qjl-quantization.md) | `quantize.rs` | Sign-based hashing with outlier separation |
| 3 | [Score Computation](03-score-computation.md) | `score.rs` | Float×sign inner product estimator |
| 4 | [Outlier Detection](04-outlier-detection.md) | `outliers.rs` | Top-k dimension norms |
| 5 | [Value Quantization](05-value-quantization.md) | `values.rs` | Min-max scalar quantization + bit-packing |
| 6 | [Fused Dequant + Dot](06-fused-dequant-dot.md) | `values.rs` | Quantized weighted sum |
| 7 | [Lloyd-Max Codebook](07-lloyd-max-codebook.md) | `codebook.rs`, `math.rs` | Optimal scalar quantization for sphere marginals |
| 8 | [Compressed Scoring](08-compressed-scoring.md) | `score.rs` | Hamming-based cosine estimator |
| 9 | [Streaming Quantizer](09-streaming-quantizer.md) | `quantizer.rs` | Batch + one-at-a-time compression |

## Cross-cutting

- [Data Structures](data-structures.md)
- [Quality Characteristics](quality.md)
- [Dependencies](dependencies.md)
