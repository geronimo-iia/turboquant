use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use qjl_sketch::outliers::detect_outliers;
use qjl_sketch::sketch::QJLSketch;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};

fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let normal: StandardNormal = StandardNormal;
    (0..d)
        .map(|_| {
            let v: f64 = normal.sample(rng);
            v as f32
        })
        .collect()
}

fn bench_score(c: &mut Criterion) {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, 64, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let query = random_vec(d, &mut rng);

    let mut group = c.benchmark_group("score_latency");

    for num_pages in [10, 100, 1000] {
        // Pre-compress all pages
        let mut all_compressed = Vec::new();
        for _ in 0..num_pages {
            let keys = random_vec(32 * d, &mut rng); // 32 vectors per page
            let outlier_indices = detect_outliers(&keys, 32, d, 4).unwrap();
            let compressed = sketch.quantize(&keys, 32, &outlier_indices).unwrap();
            all_compressed.push(compressed);
        }

        group.bench_with_input(BenchmarkId::new("pages", num_pages), &num_pages, |b, _| {
            b.iter(|| {
                let mut total_score = 0.0f32;
                for compressed in &all_compressed {
                    let scores = sketch
                        .score(black_box(&query), black_box(compressed))
                        .unwrap();
                    total_score += scores.iter().sum::<f32>();
                }
                black_box(total_score)
            });
        });
    }

    group.finish();
}

fn bench_score_single(c: &mut Criterion) {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, 64, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let query = random_vec(d, &mut rng);

    let keys = random_vec(64 * d, &mut rng);
    let outlier_indices = detect_outliers(&keys, 64, d, 4).unwrap();
    let compressed = sketch.quantize(&keys, 64, &outlier_indices).unwrap();

    c.bench_function("score_single_page_64vec", |b| {
        b.iter(|| sketch.score(black_box(&query), black_box(&compressed)));
    });
}

criterion_group!(benches, bench_score, bench_score_single);
criterion_main!(benches);
