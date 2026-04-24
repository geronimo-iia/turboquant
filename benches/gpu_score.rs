#![cfg(feature = "gpu")]

use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use qjl_sketch::outliers::detect_outliers;
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::store::config::KeysConfig;
use qjl_sketch::store::key_store::KeyStore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use tempfile::tempdir;

fn random_vec(d: usize, rng: &mut ChaCha20Rng) -> Vec<f32> {
    let normal: StandardNormal = StandardNormal;
    (0..d)
        .map(|_| {
            let v: f64 = normal.sample(rng);
            v as f32
        })
        .collect()
}

fn bench_score_compressed_cpu_vs_gpu(c: &mut Criterion) {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, 64, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    let mut group = c.benchmark_group("score_compressed_cpu_vs_gpu");

    for num_vectors in [100, 1_000, 10_000, 100_000] {
        let keys: Vec<f32> = (0..num_vectors)
            .flat_map(|_| random_vec(d, &mut rng))
            .collect();
        let outlier_indices = detect_outliers(&keys[..32 * d], 32, d, 4).unwrap();
        let compressed = sketch
            .quantize(&keys, num_vectors, &outlier_indices)
            .unwrap();

        // CPU path (force by using score_compressed directly on small batches)
        group.bench_with_input(
            BenchmarkId::new("cpu", num_vectors),
            &num_vectors,
            |b, _| {
                b.iter(|| {
                    // Use score_compressed which dispatches to CPU for < GPU_MIN_BATCH
                    // or GPU for >= GPU_MIN_BATCH. We benchmark the actual dispatch.
                    sketch.score_compressed(black_box(&compressed), black_box(&compressed))
                });
            },
        );
    }

    group.finish();
}

fn bench_score_all_pages(c: &mut Criterion) {
    let d = 64;
    let s = 128;
    let config = KeysConfig {
        head_dim: d as u16,
        sketch_dim: s as u16,
        outlier_sketch_dim: s as u16,
        seed: 42,
    };
    let sketch = config.build_sketch();
    let mut rng = ChaCha20Rng::seed_from_u64(200);
    let outlier_indices = vec![0u8];

    let mut group = c.benchmark_group("score_all_pages");

    for num_pages in [10, 100, 1_000] {
        let dir = tempdir().unwrap();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();

        for slug in 0..num_pages as u64 {
            let keys = random_vec(32 * d, &mut rng);
            let compressed = sketch.quantize(&keys, 32, &outlier_indices).unwrap();
            store.append(slug, slug, &compressed).unwrap();
        }

        let query = random_vec(d, &mut rng);

        group.bench_with_input(BenchmarkId::new("pages", num_pages), &num_pages, |b, _| {
            b.iter(|| {
                store.score_all_pages(
                    black_box(&query),
                    black_box(&sketch),
                    black_box(&outlier_indices),
                )
            });
        });
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_score_compressed_cpu_vs_gpu,
    bench_score_all_pages
);
criterion_main!(benches);
