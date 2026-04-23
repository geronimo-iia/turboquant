use criterion::{black_box, criterion_group, criterion_main, BenchmarkId, Criterion};
use qjl_sketch::outliers::detect_outliers;
use qjl_sketch::sketch::QJLSketch;
use qjl_sketch::values::quantize_values;
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

fn bench_key_quantize(c: &mut Criterion) {
    let d = 128;
    let s = 256;
    let sketch = QJLSketch::new(d, s, 64, 42).unwrap();
    let mut rng = ChaCha20Rng::seed_from_u64(100);

    let mut group = c.benchmark_group("key_quantize");

    for num_vectors in [32, 128, 512] {
        let keys = random_vec(num_vectors * d, &mut rng);
        let outlier_indices = detect_outliers(&keys, num_vectors, d, 4).unwrap();

        group.bench_with_input(
            BenchmarkId::new("vectors", num_vectors),
            &num_vectors,
            |b, _| {
                b.iter(|| {
                    sketch.quantize(
                        black_box(&keys),
                        black_box(num_vectors),
                        black_box(&outlier_indices),
                    )
                });
            },
        );
    }

    group.finish();
}

fn bench_value_quantize(c: &mut Criterion) {
    let mut rng = ChaCha20Rng::seed_from_u64(200);

    let mut group = c.benchmark_group("value_quantize");

    for num_elements in [256, 1024, 4096] {
        let values = random_vec(num_elements, &mut rng);

        group.bench_with_input(
            BenchmarkId::new("elements_4bit", num_elements),
            &num_elements,
            |b, _| {
                b.iter(|| quantize_values(black_box(&values), 32, 4));
            },
        );
    }

    group.finish();
}

fn bench_sketch_creation(c: &mut Criterion) {
    let mut group = c.benchmark_group("sketch_creation");

    for (d, s) in [(64, 128), (128, 256), (128, 512)] {
        group.bench_with_input(
            BenchmarkId::new("dim", format!("{d}x{s}")),
            &(d, s),
            |b, &(d, s)| {
                b.iter(|| {
                    QJLSketch::new(black_box(d), black_box(s), black_box(s / 4), 42).unwrap()
                });
            },
        );
    }

    group.finish();
}

criterion_group!(
    benches,
    bench_key_quantize,
    bench_value_quantize,
    bench_sketch_creation
);
criterion_main!(benches);
