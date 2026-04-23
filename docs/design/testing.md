# TurboQuant — Test Strategy

## Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 4: End-to-end (pipeline)                 │  Phase 4
│  "search query → ranked pages with scores"      │
├─────────────────────────────────────────────────┤
│  Layer 3: Persistence (store round-trip)        │  Phase 3
│  "compress → write → reload → score = same"     │
├─────────────────────────────────────────────────┤
│  Layer 2: Quality (statistical validation)      │  Phase 2 ✓
│  "our scores ≈ exact dot products"              │
├─────────────────────────────────────────────────┤
│  Layer 1: Unit (each algorithm in isolation)    │  Phase 1 ✓
│  "rotation preserves norm, popcount is correct" │
└─────────────────────────────────────────────────┘
```

## Layer 1 — Unit Tests (32 tests) ✓

Inline in each `src/*.rs` module via `#[cfg(test)]`.

### sketch.rs (6 tests)

| Test | What it checks |
|------|----------------|
| `test_proj_dimensions` | Output shapes match (d,s) and (s,d) |
| `test_proj_orthogonality` | Q^T Q ≈ d·I for first d×d chunk |
| `test_proj_deterministic` | Same seed → same matrices |
| `test_proj_different_seeds` | Different seed → different matrices |
| `test_transpose_roundtrip` | transpose(transpose(M)) == M |
| `test_matvec` | Known 2×3 matrix times 3-vector |

### outliers.rs (4 tests)

| Test | What it checks |
|------|----------------|
| `test_outlier_known_spike` | Dim with 100x magnitude is detected |
| `test_outlier_count_respected` | Returns exactly `count` indices |
| `test_outlier_multiple` | Multiple outlier dims both detected |
| `test_outlier_mask` | Mask correctly marks outlier positions |

### quantize.rs (5 tests)

| Test | What it checks |
|------|----------------|
| `test_pack_unpack_roundtrip` | pack(unpack(x)) == x |
| `test_pack_known_byte` | [1,0,1,1,0,0,1,0] → 0x4D |
| `test_quantize_output_shape` | Compressed arrays have correct sizes |
| `test_quantize_norms` | Full norm and outlier norm are correct |
| `test_quantize_outlier_separation` | Outlier energy isolated correctly |

### score.rs (5 tests)

| Test | What it checks |
|------|----------------|
| `test_signed_dot_all_positive` | All +1 signs → sum of query sketch |
| `test_signed_dot_all_negative` | All -1 signs → negative sum |
| `test_score_identical_vectors` | score(v, compress(v)) ≈ dot(v,v) within 35% |
| `test_score_sign_preserved` | sign(score) == sign(exact dot product) |
| `test_score_multiple_vectors` | 10 vectors scored, all finite |

### values.rs (6 tests)

| Test | What it checks |
|------|----------------|
| `test_quantize_4bit_round_trip` | Dequantized values within scale/2 of original |
| `test_quantize_2bit_round_trip` | Same for 2-bit |
| `test_quantize_4bit_range` | All quantized values in [0, 15] |
| `test_quantize_2bit_range` | All quantized values in [0, 3] |
| `test_quantized_dot` | Fused dot ≈ exact dot within 5% |
| `test_quantized_dot_weighted` | Weighted dot with sparse weights |

### quantizer.rs (6 tests)

| Test | What it checks |
|------|----------------|
| `test_build_sketch` | 16 vectors → 16 compressed, no residual |
| `test_build_sketch_with_remainder` | 10 vectors → 8 compressed + 2 residual |
| `test_stream_residual_buffer` | 5 tokens → all in residual, nothing compressed |
| `test_stream_buffer_flush` | 8 tokens (= buffer_size) → flushed, residual empty |
| `test_stream_matches_batch` | Stream and batch produce close scores |
| `test_attention_score_length` | 12 tokens → 12 scores (8 compressed + 4 residual) |

## Layer 2 — Quality Tests (9 tests) ✓

Integration tests in `tests/quality/`. Run with `cargo test --test quality`.
All run by default (no `#[ignore]`). Total runtime ~7 seconds.

### test_rotation.rs (2 tests)

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_rotation_preserves_norm` | ratio ∈ [0.90, 1.10] | `\|\|proj@v\|\| / (√s · \|\|v\|\|)` ≈ 1.0 over 1K vectors |
| `test_rotation_preserves_inner_product` | mean error < 0.15 | `proj_q · proj_k / s` ≈ `q · k` over 1K pairs |

### test_distortion.rs (2 tests)

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_distortion_rate` | < 0.35 | MSE/signal at s=2d over 10K pairs |
| `test_distortion_decreases_with_sketch_dim` | d1 > d2 > d4 | Monotonic improvement |

### test_ranking.rs (2 tests)

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_top_k_recall` | ≥ 0.55 mean | Top-10 overlap, 200 keys, 100 trials |
| `test_kendall_tau` | > 0.70 mean | Rank correlation, 100 keys, 50 trials |

### test_value_accuracy.rs (2 tests)

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_value_quantized_matmul_error_4bit` | < 0.20 | Mean relative error, 1K trials |
| `test_value_quantized_matmul_error_2bit` | < 1.0 | Mean relative error, 1K trials |

### test_outlier_benefit.rs (1 test)

| Test | Threshold | What it validates |
|------|-----------|-------------------|
| `test_outlier_vs_no_outlier` | ≥ 20% reduction | Distortion with vs without outlier separation |

## Layer 3 — Persistence Tests (planned)

Will live in `tests/persistence/`. Covers:

- Store write → reload → byte-identical compressed data
- Score survives persistence (compress, store, reload, score = same)
- Append-only update with generation tracking
- Dead space tracking
- Compaction reclaims space, preserves scores
- Crash recovery: truncated tail, stale index

## Layer 4 — End-to-End Tests (planned)

Will live in `tests/e2e/`. Covers:

- Pipeline smoke test: compress pages → query → ranked results
- Relevant page ranks above irrelevant
- Incremental page update changes ranking
- Edge cases: empty store, single token, no match

## Test Layout

```
src/
├── sketch.rs           ← 6 unit tests
├── outliers.rs         ← 4 unit tests
├── quantize.rs         ← 5 unit tests
├── score.rs            ← 5 unit tests
├── values.rs           ← 6 unit tests
└── quantizer.rs        ← 6 unit tests

tests/
├── quality/
│   ├── main.rs
│   ├── helpers.rs
│   ├── test_rotation.rs        ← 2 tests
│   ├── test_distortion.rs      ← 2 tests
│   ├── test_ranking.rs         ← 2 tests
│   ├── test_value_accuracy.rs  ← 2 tests
│   └── test_outlier_benefit.rs ← 1 test
├── persistence/                ← Phase 3
└── e2e/                        ← Phase 4
```

## Running

```bash
cargo test                       # all tests (unit + quality)
cargo test --lib                 # unit tests only (~0.2s)
cargo test --test quality        # quality tests only (~7s)
cargo clippy -- -D warnings      # lint
cargo fmt -- --check             # format check
```
