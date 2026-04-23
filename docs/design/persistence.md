# TurboQuant — Persistence & Loading

## What Gets Persisted

Two independent stores: **keys** (for scoring) and **values** (for
full attention output). Each store has its own data file and index.

### Key store

One entry per page. Maps to `CompressedKeys` from `src/quantize.rs`.

| Field | Type | Size (500 tokens, s=256, os=64) |
|-------|------|---------------------------------|
| key_quant | [n × s/8] u8 | 16 KB |
| key_outlier_quant | [n × os/8] u8 | 4 KB |
| key_norms | [n] f32 | 2 KB |
| outlier_norms | [n] f32 | 2 KB |
| outlier_indices | [outlier_count] u8 | ~7 B |

~24 KB per page at 500 tokens.

### Value store

One entry per page. Maps to `CompressedValues` from `src/values.rs`.

| Field | Type | Size (500 tokens, 4-bit, group=32) |
|-------|------|------------------------------------|
| packed | [n / feat_per_int] i32 | ~1 KB |
| scale | [num_groups] f32 | ~64 B |
| mn | [num_groups] f32 | ~64 B |

~1.2 KB per page at 500 tokens, 4-bit.

## File Layout

```
<store-dir>/
├── keys.bin        ← append-only CompressedKeys entries
├── keys.idx        ← sketch params + slug_hash → (offset, len, gen)
├── values.bin      ← append-only CompressedValues entries
└── values.idx      ← value params + slug_hash → (offset, len, gen)
```

No `config.bin`. Sketch parameters live in `keys.idx` header. Value
parameters live in `values.idx` header. The projection matrix is
recomputed from the seed on open (deterministic, milliseconds).

## keys.idx

```
Header (32 bytes):
Offset  Size     Field
0       4        magic: b"TQKI"
4       2        version: u16 (1)
6       2        head_dim: u16
8       2        sketch_dim: u16
10      2        outlier_sketch_dim: u16
12      8        seed: u64
20      2        entry_count: u16
22      2        padding
24      4        live_bytes: u32
28      4        dead_bytes: u32

Entries (sorted by slug_hash, 32 bytes each):
0       8        slug_hash: u64
8       8        offset: u64 (into keys.bin)
16      4        entry_len: u32
20      4        generation: u32
24      8        content_hash: u64
```

On open: read header → `QJLSketch::new(head_dim, sketch_dim,
outlier_sketch_dim, seed)` → mmap `keys.bin` → ready to score.

## values.idx

```
Header (24 bytes):
Offset  Size     Field
0       4        magic: b"TQVI"
4       2        version: u16 (1)
6       1        bits: u8 (2 or 4)
7       1        padding
8       2        group_size: u16
10      2        entry_count: u16
12      4        live_bytes: u32
16      4        dead_bytes: u32
20      4        padding

Entries (sorted by slug_hash, 32 bytes each):
0       8        slug_hash: u64
8       8        offset: u64 (into values.bin)
16      4        entry_len: u32
20      4        generation: u32
24      8        content_hash: u64
```

## keys.bin — entry format

```
Offset  Size     Field
0       4        magic: b"TQKE"
4       4        entry_len: u32
8       8        slug_hash: u64
16      4        num_vectors: u32
20      1        outlier_count: u8
21      3        padding
24      ...      outlier_indices: [outlier_count] u8
...     ...      key_quant: [num_vectors × sketch_dim/8] u8
...     ...      key_outlier_quant: [num_vectors × outlier_sketch_dim/8] u8
...     ...      key_norms: [num_vectors] f32 LE
...     ...      outlier_norms: [num_vectors] f32 LE
```

Section sizes computable from `num_vectors`, `outlier_count`, and
index header params.

## values.bin — entry format

```
Offset  Size     Field
0       4        magic: b"TQVE"
4       4        entry_len: u32
8       8        slug_hash: u64
16      4        num_elements: u32
20      4        num_groups: u32
24      ...      packed: [num_elements / feat_per_int] i32 LE
...     ...      scale: [num_groups] f32 LE
...     ...      mn: [num_groups] f32 LE
```

`feat_per_int = 32 / bits` from the index header.

## Loading

```rust
pub struct KeyStore {
    sketch: QJLSketch,
    data: Mmap,                   // keys.bin
    index: Vec<IndexEntry>,       // from keys.idx
}

pub struct ValueStore {
    bits: u8,
    group_size: usize,
    data: Mmap,                   // values.bin
    index: Vec<IndexEntry>,       // from values.idx
}
```

Score-only: open `KeyStore`. Full attention: open both.

**Cold start (score-only):** read keys.idx header (~32 bytes),
construct QJLSketch (milliseconds), load index entries into Vec,
mmap keys.bin. Two syscalls + one small allocation.

## Write Path

```
compress_page(keys, values, slug, content_hash):
  1. Detect outliers, quantize keys → CompressedKeys
  2. Quantize values → CompressedValues
  3. Serialize key entry → append to keys.bin, fsync
  4. Serialize value entry → append to values.bin, fsync
  5. Update keys.idx atomically (.tmp → rename)
  6. Update values.idx atomically (.tmp → rename)
```

Crash between step 5 and 6: keys updated, values stale. On next
open, `content_hash` mismatch in values.idx triggers re-compress
of values only. Keys are fine.

## Staleness

Each index entry stores `content_hash`. Check independently:

```rust
impl KeyStore {
    pub fn is_fresh(&self, slug_hash: u64, content_hash: u64) -> bool { ... }
}
impl ValueStore {
    pub fn is_fresh(&self, slug_hash: u64, content_hash: u64) -> bool { ... }
}
```

Keys fresh but values stale? Score works, re-compress values in
background. Both stale? Re-compress both.

## Compaction

Independent per store. Keys may churn faster than values (different
sketch_dim experiments). Each store compacts when
`dead_bytes > live_bytes / 2`.

```
compact(store.bin, store.idx):
  1. Create store.bin.new
  2. Copy live entries, record new offsets
  3. fsync store.bin.new
  4. Build store.idx.new
  5. fsync store.idx.new
  6. rename store.idx.new → store.idx
  7. rename store.bin.new → store.bin
```

## Crash Recovery

Per store, independently:

- **Truncated tail:** magic check on last entry, truncate if invalid.
- **Index missing:** rebuild by scanning .bin entries.
- **Index ahead of .bin:** entry offset beyond file length → drop
  from index, rebuild.

## Concurrency

| Operation | Lock | Impact on readers |
|-----------|------|-------------------|
| Score query | None | mmap keys.bin read-only |
| Full attention | None | mmap both read-only |
| Append keys | Keys mutex | pwrite at EOF |
| Append values | Values mutex | pwrite at EOF |
| Compact keys | Keys exclusive | readers on old mmap |
| Compact values | Values exclusive | readers on old mmap |

No lock contention between key and value operations.

## Size Estimates

| Pages | Tokens/page | keys.bin | keys.idx | values.bin | values.idx |
|-------|-------------|----------|----------|------------|------------|
| 100 | 500 | ~2.4 MB | ~3 KB | ~120 KB | ~3 KB |
| 1,000 | 500 | ~24 MB | ~32 KB | ~1.2 MB | ~32 KB |
| 10,000 | 500 | ~240 MB | ~320 KB | ~12 MB | ~320 KB |
