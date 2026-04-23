# TurboQuant Study

## Context

Source: [KV Cache Is Eating Your VRAM. Here's How Google Fixed It With TurboQuant](https://towardsdatascience.com/kv-cache-is-eating-your-vram-heres-how-google-fixed-it-with-turboquant/)
Paper: Zandieh et al. (2025). *TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate.* arXiv:2504.19874

## What TurboQuant Is

A two-stage KV cache compression framework that achieves 4.5–5x compression
(3.5–2.5 bits per channel) with near-zero accuracy loss. Proven to sit at
the theoretical optimum for its bit budget.

### Stage 1 — PolarQuant

1. **Randomized rotation** (y = R·x, R orthogonal) spreads outlier energy
   across all dimensions → isotropic distribution. Each coordinate follows
   Beta(1/2, (d−1)/2).
2. **Lloyd-Max quantization** with a precomputed codebook (depends only on
   head dimension d and bit-width b). Stores indexes (b−1 bits per dim).
3. Dequantize: codebook lookup → inverse rotation → K̂.
4. Residual: ε = K − K̂.

### Stage 2 — Residual Correction (QJL)

1. Random projection: sign(ε · S), where S is (d, d) random matrix.
   Johnson-Lindenstrauss lemma guarantees inner-product preservation.
2. Store: sign bits (1 bit per dim) + L2 norm ‖ε‖₂ (one scalar).
3. Reconstruct: K̃_QJL = (√(π/2) / d) · ‖ε‖₂ · Sᵀ · QJL.
4. Final: K̃ = K̂ + K̃_QJL.

### What sits in cache per token

| Component | Size | Purpose |
|-----------|------|---------|
| idx | b−1 bits × d | Lloyd-Max codebook indexes |
| QJL signs | 1 bit × d | Residual direction |
| ‖ε‖₂ | 1 scalar | Residual magnitude |

Total: b bits per dimension + negligible scalar overhead.

## References

1. Zandieh et al. (2025). TurboQuant: Online Vector Quantization with Near-optimal Distortion Rate. arXiv:2504.19874
2. Zandieh et al. (2024). QJL: 1-Bit Quantized JL Transform for KV Cache Quantization with Zero Overhead. AAAI 2025
3. Vaswani et al. (2017). Attention Is All You Need. NeurIPS 2017
4. Kwon et al. (2023). Efficient Memory Management for Large Language Model Serving with PagedAttention. SOSP 2023
