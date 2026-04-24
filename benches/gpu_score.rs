#![cfg(feature = "gpu")]

use criterion::{criterion_group, criterion_main, BenchmarkId, Criterion};
use qjl_sketch::store::config::KeysConfig;
use qjl_sketch::store::key_store::KeyStore;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use rand_distr::{Distribution, StandardNormal};
use std::hint::black_box;
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

const VECS_PER_PAGE: usize = 32;

struct BenchConfig {
    d: usize,
    s: usize,
    label: &'static str,
}

const CONFIGS: &[BenchConfig] = &[
    BenchConfig {
        d: 64,
        s: 128,
        label: "d64",
    },
    BenchConfig {
        d: 128,
        s: 256,
        label: "d128",
    },
];

const PAGE_COUNTS: &[usize] = &[10, 100, 1_000, 10_000];

fn make_store(
    d: usize,
    s: usize,
    num_pages: usize,
    rng: &mut ChaCha20Rng,
) -> (KeyStore, KeysConfig, tempfile::TempDir) {
    let config = KeysConfig {
        head_dim: d as u16,
        sketch_dim: s as u16,
        outlier_sketch_dim: s as u16,
        seed: 42,
    };
    let sketch = config.build_sketch();
    let dir = tempdir().unwrap();
    let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
    let outlier_indices = vec![0u8];

    for slug in 0..num_pages as u64 {
        let keys = random_vec(VECS_PER_PAGE * d, rng);
        let compressed = sketch
            .quantize(&keys, VECS_PER_PAGE, &outlier_indices)
            .unwrap();
        store.append(slug, slug, &compressed).unwrap();
    }

    (store, config, dir)
}

/// CPU baseline: explicit sketch.score() per page (never GPU)
fn bench_cpu_per_page(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    for cfg in CONFIGS {
        let mut group = c.benchmark_group(format!("cpu_per_page_{}", cfg.label));

        for &num_pages in PAGE_COUNTS {
            let (store, config, _dir) = make_store(cfg.d, cfg.s, num_pages, &mut rng);
            let sketch = config.build_sketch();
            let query = random_vec(cfg.d, &mut rng);

            group.bench_with_input(BenchmarkId::new("pages", num_pages), &num_pages, |b, _| {
                b.iter(|| {
                    let mut total = 0.0f32;
                    for slug in 0..num_pages as u64 {
                        if let Some(page) = store.get_page(slug) {
                            let keys = page.to_compressed_keys(cfg.d);
                            let scores = sketch.score(black_box(&query), black_box(&keys)).unwrap();
                            total += scores.iter().sum::<f32>();
                        }
                    }
                    black_box(total)
                });
            });
        }

        group.finish();
    }
}

/// score_all_pages with auto-dispatch (GPU when >= GPU_MIN_BATCH)
fn bench_score_all_pages(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    for cfg in CONFIGS {
        let mut group = c.benchmark_group(format!("score_all_pages_{}", cfg.label));

        for &num_pages in PAGE_COUNTS {
            let (store, config, _dir) = make_store(cfg.d, cfg.s, num_pages, &mut rng);
            let sketch = config.build_sketch();
            let query = random_vec(cfg.d, &mut rng);
            let outlier_indices = vec![0u8];

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
}

criterion_group!(benches, bench_cpu_per_page, bench_score_all_pages);
criterion_main!(benches);
