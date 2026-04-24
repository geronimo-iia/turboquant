# Algorithm 8: Compressed-vs-Compressed Scoring

Source: sign-random-projection cosine estimator (Charikar 2002 / SimHash).
Impl: `src/score.rs` → `QJLSketch::score_compressed`, `score_compressed_pair`,
`hamming_similarity`.

Estimates inner product between two compressed vectors using Hamming
distance on their packed sign bits. Less accurate than float×sign
(`score`) but requires no float projection — pure byte XOR + popcount.

```
Input:
  a, b : CompressedKeys (from Algorithm 2, same sketch parameters)

Output:
  scores : [num_vectors] f32  (one per vector pair)

Algorithm:
  For each vector pair (a[i], b[i]):

  1. Inlier score:
       sim = hamming_similarity(a.key_quant[i], b.key_quant[i], sketch_dim)
       cos_inlier = cos(π · (1 - sim))
       inlier_norm_a = sqrt(a.key_norms[i]² - a.outlier_norms[i]²)
       inlier_norm_b = sqrt(b.key_norms[i]² - b.outlier_norms[i]²)
       score_inlier = inlier_norm_a · inlier_norm_b · cos_inlier

  2. Outlier score:
       sim_o = hamming_similarity(a.key_outlier_quant[i], b.key_outlier_quant[i], os)
       cos_outlier = cos(π · (1 - sim_o))
       score_outlier = a.outlier_norms[i] · b.outlier_norms[i] · cos_outlier

  3. Total:
       score = score_inlier + score_outlier

hamming_similarity(a, b, total_bits):
  matching = popcount(NOT(a XOR b)) - padding_bits
  return matching / total_bits
```

The cosine estimator comes from sign-random-projection theory:
P(sign match) = 1 - θ/π, so θ = π·(1 - fraction_matching).

`score_compressed_pair(a, i, b, j)` extracts single vectors from
different-sized sets for cross-comparison (e.g. page-to-page similarity).
