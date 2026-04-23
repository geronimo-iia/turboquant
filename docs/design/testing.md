# TurboQuant — Test Strategy

## Layers

```
┌─────────────────────────────────────────────────┐
│  Layer 4: End-to-end (pipeline)                 │
│  "search query → ranked pages with scores"      │
├─────────────────────────────────────────────────┤
│  Layer 3: Persistence (store round-trip)        │
│  "compress → write → reload → score = same"     │
├─────────────────────────────────────────────────┤
│  Layer 2: Quality (accuracy vs. reference)      │
│  "our scores ≈ exact dot products"              │
├─────────────────────────────────────────────────┤
│  Layer 1: Unit (each algorithm in isolation)    │
│  "rotation preserves norm, popcount is correct" │
└─────────────────────────────────────────────────┘
```

## Layer 1 — Unit Tests

One test module per algorithm from `algorithms.md`.

### 1.1 Random projection matrix

```
test_proj_dimensions
  QJLSketch::new(128, 256, seed=42)
  assert proj_dir_score.shape == (128, 256)
  assert proj_dir_quant.shape == (256, 128)

test_proj_orthogonality
  // Each d×d chunk of proj_dir_score should be approximately orthogonal
  chunk = proj_dir_score[:, 0..128]
  assert (chunk^T · chunk - I·d).max_abs() < 1e-4

test_proj_deterministic
  a = QJLSketch::new(128, 256, seed=42)
  b = QJLSketch::new(128, 256, seed=42)
  assert a.proj_dir_score == b.proj_dir_score

test_proj_different_seeds
  a = QJLSketch::new(128, 256, seed=42)
  b = QJLSketch::new(128, 256, seed=99)
  assert a.proj_dir_score != b.proj_dir_score
```

### 1.2 QJL quantization (sign hashing)

```
test_quant_output_shape
  keys = randn(1, 1, 1, 32, 128)  // 1 batch, 1 head, 1 group, 32 tokens, d=128
  sketch_dim = 256
  result = sketch.quantize(keys, outlier_indices)
  assert result.key_quant.shape == (1, 1, 1, 32, 256/8)  // 32 bytes per token

test_quant_bit_packing
  // Known input: projection of a single vector gives known signs
  // Verify the packed byte matches manual bit assembly
  signs = [1, 0, 1, 1, 0, 0, 1, 0]  // expected from a known vector
  expected_byte = 0b01001101  // = 0x4D
  assert packed_byte == expected_byte

test_quant_outlier_separation
  // Set outlier_indices = [0, 1]
  // Verify key_quant ignores dims 0,1
  // Verify key_outlier_quant only uses dims 0,1

test_quant_norms
  keys = known_vector
  outlier_indices = [3, 7]
  result = sketch.quantize(keys, outlier_indices)
  expected_norm = sqrt(keys[3]^2 + keys[7]^2)
  assert (result.outlier_norms - expected_norm).abs() < 1e-6
```

### 1.3 Score computation

```
test_score_identical_vectors
  // q == k → score should be maximal (close to ||q||·||k||)
  q = randn(1, 1, 1, 128)
  compressed_k = sketch.quantize(q.unsqueeze(2))
  score = sketch.score(q, compressed_k)
  exact = dot(q, q)
  assert (score - exact).abs() / exact < 0.1  // within 10%

test_score_orthogonal_vectors
  // q ⊥ k → score should be near zero
  q = [1, 0, 0, ..., 0]
  k = [0, 1, 0, ..., 0]
  score = sketch.score(q, compress(k))
  assert score.abs() < 0.1 * norm(q) * norm(k)

test_score_sign_preserved
  // q · k > 0 → score > 0
  // q · k < 0 → score < 0
  q = randn(128)
  k = q + small_noise  // positive dot product
  assert sketch.score(q, compress(k)) > 0
  assert sketch.score(q, compress(-k)) < 0

test_popcount_correctness
  // Exhaustive for u8: all 256 values
  for byte in 0..=255u8:
    assert byte.count_ones() == naive_popcount(byte)
```

### 1.4 Outlier detection

```
test_outlier_top_k
  keys = zeros(1, 1, 1, 8, 4)  // 8 tokens, d=4
  keys[..., 2] = 100.0  // dim 2 is the outlier
  indices = detect_outliers(keys, count=1)
  assert indices == [2]

test_outlier_count_respected
  indices = detect_outliers(keys, count=3)
  assert indices.len() == 3
```

### 1.5 Value quantization

```
test_value_round_trip
  values = randn(1, 1, 128, 64)
  packed, scale, mn = quantize_values(values, group_size=32, bits=4)
  reconstructed = dequantize(packed, scale, mn, bits=4)
  // 4-bit: max error = scale / 2 per element
  assert (values - reconstructed).abs().max() < scale.max() / 2 + 1e-6

test_value_4bit_range
  // All quantized values should be in [0, 15]
  packed, _, _ = quantize_values(values, group_size=32, bits=4)
  for each unpacked value: assert 0 <= val <= 15

test_value_2bit_range
  packed, _, _ = quantize_values(values, group_size=32, bits=2)
  for each unpacked value: assert 0 <= val <= 3

test_quantized_matmul
  weights = randn(1, 1, 1, 64)  // attention weights
  values = randn(1, 1, 128, 64)
  exact = weights @ values.T
  packed, scale, mn = quantize_values(values, 32, 4)
  approx = quantized_matmul(weights, packed, scale, mn, 4)
  assert relative_error(exact, approx) < 0.05
```

### 1.6 Streaming update

```
test_stream_matches_batch
  keys = randn(1, 1, 64, 128)  // 64 tokens
  // Batch: compress all at once
  batch_result = quantizer.build_sketch(keys)
  // Stream: feed one token at a time
  stream_quantizer = KeyQuantizer::new(...)
  for t in 0..64:
    stream_quantizer.update(keys[:, :, t:t+1, :])
  // Scores should match
  q = randn(1, 1, 1, 128)
  assert batch_result.score(q) ≈ stream_quantizer.score(q)

test_stream_residual_buffer
  // Feed 10 tokens with buffer_size=32
  // Verify residual holds 10 tokens, no quantization yet
  quantizer.update(token) × 10
  assert quantizer.residual.len() == 10
  assert quantizer.key_quant.is_empty()

test_stream_buffer_flush
  // Feed 32 tokens with buffer_size=32
  // Verify residual is empty, quantized state exists
  quantizer.update(token) × 32
  assert quantizer.residual.is_none()
  assert quantizer.seq_len == 32
```

## Layer 2 — Quality Tests

Compare our Rust implementation against the Python reference and
against exact (uncompressed) computation.

### 2.1 Cross-validation with Python reference

```
test_match_python_qjl
  // Generate test data with known seed
  // Run Python QJL: save key_quant, norms, scores to .npz
  // Run Rust with same seed and input
  // Compare byte-for-byte (quantized) and within tolerance (scores)

  python_scores = load("reference/scores_seed42.npz")
  rust_scores = our_implementation(same_input, seed=42)
  assert max_abs_diff(python_scores, rust_scores) < 1e-5
```

Ship reference `.npz` files in `tests/fixtures/`. Generated once from
the Python QJL repo with pinned seeds.

### 2.2 Distortion rate

```
test_distortion_rate
  // The paper claims near-optimal distortion
  // Measure: E[||q·k - q·k̃||²] / E[||q·k||²]
  // Over 10K random (q, k) pairs

  d = 128, sketch_dim = 256
  pairs = 10_000
  exact_dots = []
  approx_dots = []
  for _ in 0..pairs:
    q, k = randn(d), randn(d)
    exact_dots.push(dot(q, k))
    approx_dots.push(sketch.score(q, compress(k)))

  mse = mean((exact - approx)^2)
  signal = mean(exact^2)
  distortion = mse / signal
  assert distortion < 0.15  // paper shows ~2.7x optimal, generous bound
```

### 2.3 Ranking preservation

```
test_ranking_preservation
  // The real test: does compression preserve the RANKING?
  // Generate 1 query + 100 key vectors
  // Rank by exact dot product and by compressed score
  // Measure rank correlation (Kendall's tau or Spearman)

  q = randn(128)
  keys = randn(100, 128)
  exact_ranking = argsort(keys @ q)
  compressed_ranking = argsort(sketch.score_batch(q, compress(keys)))
  tau = kendall_tau(exact_ranking, compressed_ranking)
  assert tau > 0.90  // strong rank correlation

test_top_k_recall
  // More practical: does top-10 by compressed score overlap with
  // top-10 by exact score?
  exact_top10 = top_k(exact_scores, 10)
  compressed_top10 = top_k(compressed_scores, 10)
  recall = |intersection| / 10
  assert recall >= 0.8  // at least 8 of 10 match
```

### 2.4 Bit-width sweep

```
test_quality_vs_bits
  // Verify quality degrades gracefully with fewer bits
  for bits in [4, 3, 2, 1]:
    distortion = measure_distortion(sketch_dim = d * bits)
    println!("{bits} bits → distortion {distortion:.4}")
  // 4 bits should be near-lossless, 1 bit should be worst
  assert distortion_4bit < distortion_3bit < distortion_2bit < distortion_1bit
```

## Layer 3 — Persistence Tests

### 3.1 Round-trip

```
test_store_write_read
  // Compress a page, write to store, reload, verify identical bytes
  compressed = compress(page_tokens)
  store.append(slug_hash, content_hash, compressed)
  store.flush()

  store2 = KVStore::open(dir)
  loaded = store2.get_page(slug_hash)
  assert loaded.key_quant() == compressed.key_quant
  assert loaded.key_norms() == compressed.key_norms

test_score_survives_persistence
  // Compress, score, persist, reload, score again → same result
  score_before = sketch.score(query, compressed)
  store.append(compressed)
  loaded = store.reload().get_page(slug_hash)
  score_after = sketch.score(query, loaded)
  assert score_before == score_after  // exact, not approximate
```

### 3.2 Append-only semantics

```
test_update_overwrites_old
  store.append(slug_hash, content_hash_v1, data_v1, generation=1)
  store.append(slug_hash, content_hash_v2, data_v2, generation=2)
  loaded = store.get_page(slug_hash)
  assert loaded.content_hash == content_hash_v2
  assert loaded.generation == 2

test_dead_space_tracked
  store.append(slug_hash, v1, gen=1)  // 1000 bytes
  store.append(slug_hash, v2, gen=2)  // 1000 bytes
  assert store.dead_bytes() == 1000
  assert store.live_bytes() == 1000
```

### 3.3 Compaction

```
test_compaction_reclaims_space
  // Write 100 pages, update 50 of them
  for i in 0..100: store.append(slug(i), data(i), gen=1)
  for i in 0..50:  store.append(slug(i), data_v2(i), gen=2)
  size_before = store.file_size()
  store.compact()
  size_after = store.file_size()
  assert size_after < size_before
  // All 100 pages still readable
  for i in 0..100: assert store.get_page(slug(i)).is_some()

test_compaction_preserves_scores
  score_before = sketch.score(query, store.get_page(slug))
  store.compact()
  score_after = sketch.score(query, store.get_page(slug))
  assert score_before == score_after
```

### 3.4 Crash recovery

```
test_truncated_tail_recovery
  // Simulate crash: write partial entry at end of store.bin
  store.append(slug, data, gen=1)
  // Append 50 bytes of garbage (simulates partial write)
  file.write_at(store.len(), &[0xDE; 50])
  // Reopen should detect bad magic, truncate tail
  store2 = KVStore::open(dir)
  assert store2.get_page(slug).is_some()  // original entry intact
  assert store2.file_size() == original_size  // garbage removed

test_index_ahead_of_store
  // Index references an entry that doesn't exist in store
  // (crash after append, before index rewrite)
  // Open should rebuild index from store
  store.append(slug_a, data_a)
  // Manually add slug_b to index without writing to store
  inject_index_entry(slug_b, offset=99999)
  store2 = KVStore::open(dir)
  assert store2.get_page(slug_a).is_some()
  assert store2.get_page(slug_b).is_none()
```

### 3.5 Staleness

```
test_stale_detection
  store.append(slug, content_hash=0xAABB, data)
  assert store.is_fresh(slug, current_hash=0xAABB) == true
  assert store.is_fresh(slug, current_hash=0xCCDD) == false
```

## Layer 4 — End-to-End Tests

Full pipeline from search query to ranked results.

### 4.1 Pipeline smoke test

```
test_pipeline_returns_results
  // Setup: 10 wiki pages, compressed and stored
  // Query: a search string
  // Expect: ranked list of pages with scores > 0

  wiki = setup_test_wiki(10_pages)
  kv = compress_all_pages(wiki)
  results = pipeline.query("mixture of experts", top_k=5)
  assert results.len() <= 5
  assert results[0].score >= results[1].score  // sorted

test_pipeline_relevant_page_ranks_high
  // Page about "transformers" should rank higher than page about
  // "cooking recipes" for query "attention mechanism"
  wiki = [page("transformers", "attention is all you need..."),
          page("recipes", "how to bake a cake...")]
  kv = compress_all_pages(wiki)
  results = pipeline.query("attention mechanism", top_k=2)
  assert results[0].slug == "transformers"
```

### 4.2 Incremental update

```
test_update_page_changes_ranking
  wiki = [page_a, page_b]
  kv = compress_all_pages(wiki)
  results_before = pipeline.query("topic X")

  // Update page_b to be highly relevant to "topic X"
  update_page(page_b, content="topic X is everything...")
  kv.recompress(page_b)
  results_after = pipeline.query("topic X")

  assert results_after[0].slug == page_b.slug
```

### 4.3 Empty and edge cases

```
test_empty_store
  results = pipeline.query("anything")
  assert results.is_empty()

test_single_token_page
  wiki = [page("tiny", "hello")]
  kv = compress_all_pages(wiki)
  results = pipeline.query("hello")
  assert results.len() == 1

test_query_no_match
  wiki = [page("rust", "systems programming language")]
  results = pipeline.query("quantum physics")
  // Should still return results (with low scores), not crash
  assert results.len() >= 0
```

## Test Fixtures

```
tests/
├── fixtures/
│   ├── reference/
│   │   ├── proj_seed42.npz         ← Python-generated projection matrix
│   │   ├── quant_seed42.npz        ← Python-generated quantized keys
│   │   └── scores_seed42.npz       ← Python-generated scores
│   └── pages/
│       ├── short_page.md           ← 10 tokens
│       ├── medium_page.md          ← 500 tokens
│       └── long_page.md            ← 5000 tokens
├── unit/
│   ├── test_projection.rs
│   ├── test_quantize.rs
│   ├── test_score.rs
│   ├── test_outliers.rs
│   ├── test_values.rs
│   └── test_streaming.rs
├── quality/
│   ├── test_distortion.rs
│   ├── test_ranking.rs
│   └── test_cross_validate.rs
├── persistence/
│   ├── test_round_trip.rs
│   ├── test_compaction.rs
│   ├── test_crash_recovery.rs
│   └── test_staleness.rs
└── e2e/
    ├── test_pipeline.rs
    └── test_incremental.rs
```

## Running

```bash
cargo test                          # all layers
cargo test unit                     # layer 1 only
cargo test quality                  # layer 2 (slower, statistical)
cargo test persistence              # layer 3
cargo test e2e                      # layer 4

cargo test -- --ignored             # include slow/statistical tests
```

Quality tests (layer 2) are `#[ignore]` by default — they run 10K+
iterations and take seconds. CI runs them, local dev skips them.
