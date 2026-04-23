use turboquant::store::config::{KeysConfig, ValuesConfig};
use turboquant::store::key_store::KeyStore;
use turboquant::store::value_store::ValueStore;
use turboquant::values::{quantize_values, quantized_dot};

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

fn keys_config() -> KeysConfig {
    KeysConfig {
        head_dim: 16,
        sketch_dim: 32,
        outlier_sketch_dim: 16,
        seed: 42,
    }
}

fn values_config() -> ValuesConfig {
    ValuesConfig {
        bits: 4,
        group_size: 8,
    }
}

#[test]
fn test_keys_fresh_values_stale() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let vc = values_config();
    let mut key_store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let mut val_store = ValueStore::create(dir.path(), vc).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(100);
    let slug: u64 = 0xAA;
    let content_v1: u64 = 0x11;
    let content_v2: u64 = 0x22;

    // Write v1 to both stores
    let keys_v1 = random_vec(4 * 16, &mut rng);
    let compressed_keys_v1 = sketch.quantize(&keys_v1, 4, &[0u8]);
    key_store
        .append(slug, content_v1, &compressed_keys_v1)
        .unwrap();

    let values_v1: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let compressed_values_v1 = quantize_values(&values_v1, 8, 4);
    val_store
        .append(slug, content_v1, &compressed_values_v1)
        .unwrap();

    // Both fresh at v1
    assert!(key_store.is_fresh(slug, content_v1));
    assert!(val_store.is_fresh(slug, content_v1));

    // Update keys to v2, leave values at v1
    let keys_v2 = random_vec(4 * 16, &mut rng);
    let compressed_keys_v2 = sketch.quantize(&keys_v2, 4, &[0u8]);
    key_store
        .append(slug, content_v2, &compressed_keys_v2)
        .unwrap();

    // Keys fresh at v2, values stale
    assert!(key_store.is_fresh(slug, content_v2));
    assert!(!key_store.is_fresh(slug, content_v1));
    assert!(!val_store.is_fresh(slug, content_v2));
    assert!(val_store.is_fresh(slug, content_v1));

    // Score still works with updated keys
    let query = random_vec(16, &mut rng);
    let page = key_store.get_page(slug).unwrap();
    let reloaded = page.to_compressed_keys(kc.head_dim as usize);
    let scores = sketch.score(&query, &reloaded);
    assert_eq!(scores.len(), 4);
    assert!(scores.iter().all(|s| s.is_finite()));
}

#[test]
fn test_both_stores_independent_lifecycle() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let vc = values_config();
    let mut key_store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let mut val_store = ValueStore::create(dir.path(), vc).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(200);

    // Add 3 pages to keys, only 2 to values
    for slug in 0u64..3 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        key_store.append(slug, slug * 10, &compressed).unwrap();
    }
    for slug in 0u64..2 {
        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let compressed = quantize_values(&values, 8, 4);
        val_store.append(slug, slug * 10, &compressed).unwrap();
    }

    assert_eq!(key_store.len(), 3);
    assert_eq!(val_store.len(), 2);

    // Page 2 has keys but no values — valid state
    assert!(key_store.get_page(2).is_some());
    assert!(val_store.get_page(2).is_none());
}

#[test]
fn test_dead_bytes_tracked_after_reopen() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(300);
    let keys = random_vec(4 * 16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8]);

    store.append(0xAA, 0x11, &compressed).unwrap();
    store.append(0xAA, 0x22, &compressed).unwrap();

    let dead_before = store.dead_bytes();
    assert!(dead_before > 0);

    // Reopen and verify dead_bytes persisted
    let store2 = KeyStore::open(dir.path()).unwrap();
    assert_eq!(store2.dead_bytes(), dead_before);
    assert_eq!(store2.len(), 1);
}

#[test]
fn test_key_store_compact_reclaims_space() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(400);

    // Write 5 pages, then update 3 of them (creates dead space)
    for slug in 0u64..5 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        store.append(slug, slug * 10, &compressed).unwrap();
    }
    for slug in 0u64..3 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        store.append(slug, slug * 100, &compressed).unwrap();
    }

    assert_eq!(store.len(), 5);
    assert!(store.dead_bytes() > 0);
    let size_before = std::fs::metadata(dir.path().join("keys.bin"))
        .unwrap()
        .len();

    store.compact().unwrap();

    let size_after = std::fs::metadata(dir.path().join("keys.bin"))
        .unwrap()
        .len();
    assert!(size_after < size_before);
    assert_eq!(store.dead_bytes(), 0);
    assert_eq!(store.len(), 5);

    // All pages still readable
    for slug in 0u64..5 {
        assert!(store.get_page(slug).is_some());
    }
}

#[test]
fn test_key_store_compact_preserves_scores() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(500);
    let keys = random_vec(4 * 16, &mut rng);
    let query = random_vec(16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8]);

    store.append(0xAA, 0x11, &compressed).unwrap();
    // Update to create dead space
    store.append(0xAA, 0x22, &compressed).unwrap();

    let page_before = store.get_page(0xAA).unwrap();
    let score_before = sketch.score(
        &query,
        &page_before.to_compressed_keys(kc.head_dim as usize),
    );

    store.compact().unwrap();

    let page_after = store.get_page(0xAA).unwrap();
    let score_after = sketch.score(&query, &page_after.to_compressed_keys(kc.head_dim as usize));

    assert_eq!(score_before, score_after);
}

#[test]
fn test_value_store_compact_reclaims_space() {
    let dir = tempdir().unwrap();
    let vc = values_config();
    let mut store = ValueStore::create(dir.path(), vc).unwrap();

    for slug in 0u64..5 {
        let values: Vec<f32> = (0..8).map(|i| (slug as f32) + i as f32).collect();
        let compressed = quantize_values(&values, 8, 4);
        store.append(slug, slug * 10, &compressed).unwrap();
    }
    for slug in 0u64..3 {
        let values: Vec<f32> = (0..8).map(|i| (slug as f32) * 2.0 + i as f32).collect();
        let compressed = quantize_values(&values, 8, 4);
        store.append(slug, slug * 100, &compressed).unwrap();
    }

    assert!(store.dead_bytes() > 0);
    let size_before = std::fs::metadata(dir.path().join("values.bin"))
        .unwrap()
        .len();

    store.compact().unwrap();

    let size_after = std::fs::metadata(dir.path().join("values.bin"))
        .unwrap()
        .len();
    assert!(size_after < size_before);
    assert_eq!(store.dead_bytes(), 0);
    assert_eq!(store.len(), 5);

    for slug in 0u64..5 {
        assert!(store.get_page(slug).is_some());
    }
}

#[test]
fn test_value_store_compact_preserves_dot() {
    let dir = tempdir().unwrap();
    let vc = values_config();
    let mut store = ValueStore::create(dir.path(), vc).unwrap();

    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let weights: Vec<f32> = (0..8).map(|i| (i as f32) * 0.1).collect();
    let compressed = quantize_values(&values, 8, 4);

    store.append(0xAA, 0x11, &compressed).unwrap();
    store.append(0xAA, 0x22, &compressed).unwrap();

    let dot_before = quantized_dot(
        &weights,
        &store.get_page(0xAA).unwrap().to_compressed_values(),
    );

    store.compact().unwrap();

    let dot_after = quantized_dot(
        &weights,
        &store.get_page(0xAA).unwrap().to_compressed_values(),
    );

    assert_eq!(dot_before, dot_after);
}

#[test]
fn test_compact_survives_reopen() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(600);
    for slug in 0u64..3 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        store.append(slug, slug, &compressed).unwrap();
    }
    // Update slug 0 to create dead space
    let keys = random_vec(4 * 16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8]);
    store.append(0, 99, &compressed).unwrap();

    store.compact().unwrap();

    // Reopen after compaction
    let store2 = KeyStore::open(dir.path()).unwrap();
    assert_eq!(store2.len(), 3);
    assert_eq!(store2.dead_bytes(), 0);
    for slug in 0u64..3 {
        assert!(store2.get_page(slug).is_some());
    }
}

#[test]
fn test_truncated_tail_recovery() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(700);
    let keys = random_vec(4 * 16, &mut rng);
    let compressed = sketch.quantize(&keys, 4, &[0u8]);
    store.append(0xAA, 0xBB, &compressed).unwrap();

    let good_size = std::fs::metadata(dir.path().join("keys.bin"))
        .unwrap()
        .len();

    // Append garbage to simulate crash mid-write
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.path().join("keys.bin"))
            .unwrap();
        f.write_all(&[0xDE; 50]).unwrap();
    }

    // Reopen should truncate the garbage
    let store2 = KeyStore::open(dir.path()).unwrap();
    assert_eq!(store2.len(), 1);
    assert!(store2.get_page(0xAA).is_some());

    let recovered_size = std::fs::metadata(dir.path().join("keys.bin"))
        .unwrap()
        .len();
    assert_eq!(recovered_size, good_size);
}

#[test]
fn test_index_ahead_of_store() {
    let dir = tempdir().unwrap();
    let kc = keys_config();
    let mut store = KeyStore::create(dir.path(), kc.clone()).unwrap();
    let sketch = kc.build_sketch();

    let mut rng = ChaCha20Rng::seed_from_u64(800);

    // Write 2 pages
    for slug in 0u64..2 {
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        store.append(slug, slug * 10, &compressed).unwrap();
    }
    drop(store);

    // Truncate keys.bin to roughly half (kills second entry)
    let full_size = std::fs::metadata(dir.path().join("keys.bin"))
        .unwrap()
        .len();
    {
        let f = std::fs::File::options()
            .write(true)
            .open(dir.path().join("keys.bin"))
            .unwrap();
        f.set_len(full_size / 2).unwrap();
    }

    // Reopen — should drop entries beyond EOF
    let store2 = KeyStore::open(dir.path()).unwrap();
    // At most 1 page survives (the first one, if it fits in half)
    assert!(store2.len() <= 1);
    // Should not panic or return corrupt data
    if store2.len() == 1 {
        assert!(store2.get_page(0).is_some());
    }
}

#[test]
fn test_value_store_truncated_tail_recovery() {
    let dir = tempdir().unwrap();
    let vc = values_config();
    let mut store = ValueStore::create(dir.path(), vc).unwrap();

    let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
    let compressed = quantize_values(&values, 8, 4);
    store.append(0xAA, 0xBB, &compressed).unwrap();

    let good_size = std::fs::metadata(dir.path().join("values.bin"))
        .unwrap()
        .len();

    // Append garbage
    {
        use std::io::Write;
        let mut f = std::fs::OpenOptions::new()
            .append(true)
            .open(dir.path().join("values.bin"))
            .unwrap();
        f.write_all(&[0xDE; 50]).unwrap();
    }

    let store2 = ValueStore::open(dir.path()).unwrap();
    assert_eq!(store2.len(), 1);
    assert!(store2.get_page(0xAA).is_some());

    let recovered_size = std::fs::metadata(dir.path().join("values.bin"))
        .unwrap()
        .len();
    assert_eq!(recovered_size, good_size);
}

#[test]
fn test_value_store_index_ahead_of_store() {
    let dir = tempdir().unwrap();
    let vc = values_config();
    let mut store = ValueStore::create(dir.path(), vc).unwrap();

    for slug in 0u64..2 {
        let values: Vec<f32> = (0..8).map(|i| (slug as f32) + i as f32).collect();
        let compressed = quantize_values(&values, 8, 4);
        store.append(slug, slug * 10, &compressed).unwrap();
    }
    drop(store);

    // Truncate values.bin to roughly half
    let full_size = std::fs::metadata(dir.path().join("values.bin"))
        .unwrap()
        .len();
    {
        let f = std::fs::File::options()
            .write(true)
            .open(dir.path().join("values.bin"))
            .unwrap();
        f.set_len(full_size / 2).unwrap();
    }

    let store2 = ValueStore::open(dir.path()).unwrap();
    assert!(store2.len() <= 1);
}
