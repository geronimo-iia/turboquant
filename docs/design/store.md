# Store — Usage Guide

## Overview

Two independent append-only stores persist compressed vectors to disk:

- `KeyStore` — compressed key vectors (`CompressedKeys`) for scoring
- `ValueStore` — compressed value vectors (`CompressedValues`) for
  full attention output

Each store has its own data file (`.bin`) and index (`.idx`). They
share a directory but operate independently.

```
<store-dir>/
├── keys.bin        ← append-only key entries
├── keys.idx        ← sketch params + offset table
├── values.bin      ← append-only value entries
└── values.idx      ← value params + offset table
```

## Lifecycle

### Create

```rust
use qjl_sketch::store::config::{KeysConfig, ValuesConfig};
use qjl_sketch::store::key_store::KeyStore;
use qjl_sketch::store::value_store::ValueStore;

let dir = Path::new("/path/to/store");

let key_store = KeyStore::create(dir, KeysConfig {
    head_dim: 128,
    sketch_dim: 256,
    outlier_sketch_dim: 64,
    seed: 42,
})?;

let val_store = ValueStore::create(dir, ValuesConfig {
    bits: 4,
    group_size: 32,
})?;
```

Creates empty `.bin` and `.idx` files. The `KeysConfig` params are
stored in `keys.idx` header — the `QJLSketch` is reconstructed from
the seed on every open (deterministic, milliseconds).

### Open

```rust
let key_store = KeyStore::open(dir)?;
let val_store = ValueStore::open(dir)?;

// Sketch is available immediately
let sketch = key_store.config.build_sketch();
```

On open:
1. Read `.idx` header (config params + index entries)
2. Truncate any partial tail in `.bin` (crash recovery)
3. Drop index entries pointing beyond `.bin` length
4. mmap `.bin`

### Compress and store a page

```rust
use qjl_sketch::outliers::detect_outliers;
use qjl_sketch::values::quantize_values;

let slug_hash: u64 = blake3_hash_of_slug;
let content_hash: u64 = blake3_hash_of_page_content;

// Keys
let outlier_indices = detect_outliers(&key_vectors, group_size, head_dim, outlier_count);
let compressed_keys = sketch.quantize(&key_vectors, num_vectors, &outlier_indices);
key_store.append(slug_hash, content_hash, &compressed_keys)?;

// Values
let compressed_values = quantize_values(&value_vectors, group_size, bits);
val_store.append(slug_hash, content_hash, &compressed_values)?;
```

Each `append`:
1. Serializes the entry to bytes
2. Appends to `.bin` (fsync)
3. Updates the in-memory index (sorted by slug_hash)
4. Rewrites `.idx` atomically (write .tmp → rename)
5. Re-mmaps `.bin`

If the slug already exists, the old entry becomes dead space.

### Score a query

```rust
// Score-only — no ValueStore needed
let page = key_store.get_page(slug_hash).unwrap();
let compressed = page.to_compressed_keys(key_store.config.head_dim as usize);
let scores = sketch.score(&query_vector, &compressed);
```

`get_page` returns a `KeyPageView` — a zero-copy view into the mmap.
`to_compressed_keys` copies the data into a `CompressedKeys` struct
for use with `sketch.score()`.

### Check staleness

```rust
if !key_store.is_fresh(slug_hash, current_content_hash) {
    // Page content changed — re-compress
}

// Keys and values can be independently stale
if key_store.is_fresh(slug_hash, hash) && !val_store.is_fresh(slug_hash, hash) {
    // Keys are current, only re-compress values
}
```

### Compaction

```rust
if key_store.dead_bytes() > key_store.live_bytes() / 2 {
    key_store.compact()?;
}
// Same for val_store independently
```

Compaction rewrites only live entries to a new file, then atomic
rename. Readers on the old mmap are unaffected (POSIX fd semantics).

### Store metrics

```rust
key_store.len()         // number of pages
key_store.live_bytes()  // bytes used by current entries
key_store.dead_bytes()  // bytes reclaimable by compaction
key_store.is_empty()
```

## Error Handling

All store operations return `qjl_sketch::error::Result`. Errors are
represented by `QjlError`:

| Operation | Possible errors |
|-----------|-----------------|
| `create` | `Io` (directory creation, file write) |
| `open` | `Io` (file not found), `StoreMagicMismatch`, `StoreVersionMismatch` |
| `append` | `Io` (disk full, fsync failure) |
| `compact` | `Io` (disk full for new file) |

The stores never panic. A corrupt `.bin` file is truncated to the
last valid entry on open. A corrupt `.idx` triggers index rebuild
(KeyStore only, from `.idx.tmp` if available).

## Crash Safety

| Scenario | Recovery |
|----------|----------|
| Crash during append (partial entry) | Truncated on next open |
| Crash after .bin write, before .idx rewrite | Index entry missing, data is dead space until compaction |
| Crash during .idx rewrite | Old .idx intact (atomic rename); .idx.tmp left behind |
| Crash during compaction | Old .bin intact (atomic rename); .bin.compact left behind |
| .idx deleted | KeyStore rebuilds from .idx.tmp if present; otherwise re-create + re-ingest |

## Concurrency

| Operation | Thread safety |
|-----------|---------------|
| `get_page` (read) | Safe — mmap is read-only |
| `is_fresh` (read) | Safe — index is in-memory Vec |
| `append` (write) | Not thread-safe — caller must synchronize |
| `compact` (write) | Not thread-safe — caller must synchronize |

For concurrent access, wrap the store in a `RwLock`: readers take
shared lock, writers (append/compact) take exclusive lock.

## File Format

See [persistence.md](persistence.md) for the binary format specification.

## Store-Level Scoring

```rust
// Score a query against all pages (float x sign)
let results = key_store.score_all_pages(&query, &sketch, &outlier_indices)?;
// Returns Vec<(slug_hash, Vec<f32>)>

// With `gpu` feature: batches all vectors into single GPU dispatch
// Without GPU: sketch.score() per page on CPU
```

See [algorithms/11-gpu-scoring.md](algorithms/11-gpu-scoring.md).

## Export / Import (`serde` feature)

```rust
// Streaming export
for entry in key_store.iter_pages() {
    serde_json::to_writer(&mut file, &entry)?;
}

// Streaming import
key_store.import_entry(&entry)?;
```

See [serde.md](serde.md).

## Dependencies

| Crate | Used for |
|-------|----------|
| `memmap2` | mmap `.bin` files |
| `bytemuck` | (available but not used — unaligned reads done manually) |
| `blake3` | Content hashing (caller's responsibility) |
