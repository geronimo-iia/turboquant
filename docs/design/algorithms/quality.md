# Quality Characteristics

Tested with random Gaussian vectors, d=64–128, s=64–512.

| Metric | Value | Conditions |
|--------|-------|------------|
| Distortion (MSE/signal) | < 0.35 | s = 2d |
| Distortion monotonicity | d > 2d > 4d | confirmed |
| Top-10 recall | ≥ 0.55 mean | 200 keys, s = 4d, 100 trials |
| Kendall's tau | > 0.70 mean | 100 keys, s = 4d, 50 trials |
| Outlier benefit | ≥ 20% distortion reduction | 10x outlier magnitude |
| Value 4-bit relative error | < 0.20 mean | random Gaussian |
| Value 2-bit relative error | < 1.0 mean | random Gaussian |

Quality improves with larger sketch_dim. At s = 8d (the paper's
recommended setting), distortion drops significantly and ranking
preservation approaches exact.
