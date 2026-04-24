# Test Strategy

## Layers

```
+---------------------------------------------------+
|  Layer 4: End-to-end (pipeline)                   |  llm-wiki
|  "search query -> ranked pages with scores"       |
+---------------------------------------------------+
|  Layer 3: Persistence + Integration               |  tests/
|  "compress -> write -> reload -> score = same"    |
+---------------------------------------------------+
|  Layer 2: Quality (statistical validation)        |  tests/quality/
|  "our scores ~ exact dot products"                |
+---------------------------------------------------+
|  Layer 1: Unit (each algorithm in isolation)      |  src/*.rs
|  "rotation preserves norm, popcount is correct"   |
+---------------------------------------------------+
```

## Test counts (v0.5.0)

| Category | Count | Feature |
|----------|-------|---------|
| Unit tests (src/) | 121 | default |
| Persistence tests | 18 | default |
| Quality tests | 11 | default |
| Serde round-trip | 12 | `--features serde` |
| GPU tests | 5 | `--features gpu -- --ignored` |
| **Total** | **167** | |

## Running

```bash
cargo test                                 # 150 tests (default)
cargo test --features serde                # +12 serde tests
cargo test --features gpu                  # +5 ignored GPU tests
cargo test --features gpu -- --ignored     # run GPU tests (needs adapter)
```

## Layer 1 -- Unit Tests (121 tests)

Inline in each `src/*.rs` module via `#[cfg(test)]`.

| Module | Tests | What it covers |
|--------|-------|----------------|
| sketch.rs | 8 | Projection dimensions, orthogonality, determinism, transpose, matvec |
| outliers.rs | 6 | Spike detection, count, mask, dimension mismatch |
| quantize.rs | 5 | Pack/unpack, output shape, norms, outlier separation |
| score.rs | 11 | signed_dot, hamming_similarity, score, score_compressed, score_compressed_pair |
| values.rs | 6 | 2/4-bit round-trip, range, quantized_dot |
| quantizer.rs | 6 | Batch, streaming, buffer flush, residual |
| error.rs | 6 | Display, From, validate_finite |
| codebook.rs | 14 | 1/2/4/8-bit, symmetry, boundaries, cache, round-trip |
| math.rs | 12 | lgamma, beta_pdf, normal_icdf, sample_beta_marginal, simpson |
| rotation.rs | 7 | Orthogonality, round-trip, norm preservation, determinism |
| mse_quant.rs | 6 | Round-trip, score vs exact, MSE decreases with bits |
| store/config.rs | 6 | Config round-trip, index entry, sketch reconstruction |
| store/key_store.rs | 24 | Create, append, get, staleness, update, compact, score_all_pages |
| store/value_store.rs | 10 | Create, append, get, staleness, update, compact |

## Layer 2 -- Quality Tests (11 tests)

Integration tests in `tests/quality/`.

| File | Tests | What it validates |
|------|-------|-------------------|
| test_rotation.rs | 2 | Norm preservation, inner product preservation |
| test_distortion.rs | 2 | MSE < 0.35 at s=2d, monotonic improvement |
| test_ranking.rs | 2 | Top-10 recall >= 0.55, Kendall's tau > 0.70 |
| test_value_accuracy.rs | 2 | 4-bit error < 0.20, 2-bit error < 1.0 |
| test_outlier_benefit.rs | 1 | >= 20% distortion reduction |
| test_compressed_ranking.rs | 1 | Compressed scoring Kendall's tau > 0.50 |
| test_mse_quant.rs | 1 | MSE ranking Kendall's tau > 0.60 |

## Layer 3 -- Persistence + Integration (18 tests)

Integration tests in `tests/persistence/main.rs`.

Covers: create/open, append/get, staleness, compaction, crash
recovery (truncated tail, index ahead of store), error handling
(NaN, dimension mismatch, invalid bit width).

## Serde Tests (12 tests, `--features serde`)

Integration tests in `tests/serde.rs`.

Round-trip (JSON serialize -> deserialize -> equal) for:
Codebook, CompressedKeys, CompressedValues, MseQuantized,
KeysConfig, ValuesConfig, IndexEntry, IndexMeta, QJLSketch,
RandomRotation, KeyStore export/import, ValueStore export/import.

## GPU Tests (5 tests, `--features gpu -- --ignored`)

Inline in `src/gpu/wgpu_backend.rs` and `src/store/key_store.rs`.
Require a GPU adapter -- marked `#[ignore]`.

| Test | What it validates |
|------|-------------------|
| gpu_context_initializes | Adapter found |
| gpu_float_sign_score_matches_cpu | GPU scores match CPU within 1e-2 |
| gpu_score_empty_input | Empty input returns empty vec |
| test_score_all_pages_gpu_dispatch | GPU dispatch returns valid results |
| test_score_all_pages_gpu_results_valid | All scores finite |
