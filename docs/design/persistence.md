# TurboQuant — Persistence & Loading

## The Problem

The compressed KV store must survive process restarts. Re-projecting
every page on startup defeats the purpose. We need:

- Fast cold start (mmap, no deserialization)
- Incremental updates (re-compress one page, not the whole store)
- Portable format (no Rust-specific serialization)
- Concurrent readers (multiple queries, one writer)

## What Gets Persisted

Two categories: **immutable config** (created once per model choice)
and **per-page state** (created/updated on page ingest).

### Immutable config (per model)

| Data | Shape | Size example (d=128, s=256) |
|------|-------|-----------------------------|
| proj_dir_score | [d, s] | 128 × 256 × 4 = 128 KB |
| proj_dir_quant | [s, d] | same, transposed |
| head_dim | scalar | 4 B |
| sketch_dim | scalar | 4 B |
| outlier_sketch_dim | scalar | 4 B |
| outlier_count | scalar | 4 B |
| group_size | scalar | 4 B |
| bits (value quant) | scalar | 1 B |
| model_id | string | variable |

This is tiny (~256 KB). Written once when the model weights are chosen.

### Per-page compressed state

| Data | Shape | Size per token |
|------|-------|----------------|
| key_quant | [heads, groups, group_size, s/8] u8 | s/8 bytes |
| key_outlier_quant | [heads, groups, group_size, os/8] u8 | os/8 bytes |
| key_norms | [heads, groups, group_size] f32 | 4 bytes |
| outlier_norms | [heads, groups, group_size] f32 | 4 bytes |
| outlier_indices | [heads, groups, outlier_count] u8 | shared per group |
| value_packed | [heads, d, seq/(32/bits)] i32 | bits/32 × 4 bytes per dim |
| value_scale | [heads, d, num_groups] f32 | small |
| value_mn | [heads, d, num_groups] f32 | small |
| seq_len | scalar | 4 B |

For d=128, s=256, 1 head, 4-bit values: ~40 bytes/token compressed
vs ~1024 bytes/token uncompressed (128 dims × 2 matrices × 4 bytes).

## File Layout

Single packed file per wiki space. One mmap, all pages.

```
~/.llm-wiki/kv-store/<wiki-name>/
├── config.bin          ← projection matrices + params (immutable)
├── store.bin           ← all compressed pages, append-only
└── store.idx           ← offset table: slug-hash → (offset, len, gen)
```

### config.bin — projection matrices + params

```
Offset  Size     Field
0       4        magic: "TQKV"
4       2        version: u16
6       2        head_dim: u16
8       2        sketch_dim: u16
10      2        outlier_sketch_dim: u16
12      1        outlier_count: u8
13      1        value_bits: u8
14      2        group_size: u16
16      2        num_heads: u16
18      2        reserved
20      32       model_id: utf8 (zero-padded)
52      4        proj_score_offset: u32
56      4        proj_score_len: u32
60      4        proj_quant_offset: u32
64      4        proj_quant_len: u32
68      ...      [proj_dir_score data: f32 LE]
...     ...      [proj_dir_quant data: f32 LE]
```

### store.bin — append-only packed pages

Pages are concatenated. Each page entry is self-describing:

```
Per-entry layout:

Offset  Size     Field
0       4        magic: "TQPG"
4       4        entry_len: u32 (total bytes including header)
8       8        slug_hash: u64 (blake3 of slug, truncated)
16      8        content_hash: u64 (blake3 of page content, truncated)
24      4        generation: u32 (monotonic, for dedup)
28      2        num_heads: u16
30      4        seq_len: u32
34      4        num_groups: u32
38      2        reserved
40      ...      [key_quant bytes]
...     ...      [key_outlier_quant bytes]
...     ...      [key_norms: f32 LE]
...     ...      [outlier_norms: f32 LE]
...     ...      [outlier_indices: u8]
...     ...      [value_packed: i32 LE]
...     ...      [value_scale: f32 LE]
...     ...      [value_mn: f32 LE]
```

Section sizes are computable from `num_heads`, `seq_len`, `num_groups`,
and the config params (sketch_dim, outlier_sketch_dim, etc.). No
per-section offset table needed — just sequential layout with known
sizes.

**Append-only rule:** updates write a new entry at the end with a
higher generation number. The old entry becomes dead space. The index
always points to the latest generation.

### store.idx — offset table

```
Offset  Size     Field
0       4        magic: "TQIX"
4       2        version: u16
6       2        entry_count: u16
8       4        live_bytes: u32 (total live data in store.bin)
12      4        dead_bytes: u32 (reclaimable)
16      ...      entries[] (sorted by slug_hash for binary search)

Each entry (32 bytes):
0       8        slug_hash: u64
8       8        offset: u64 (byte offset into store.bin)
16      4        entry_len: u32
20      4        generation: u32
24      8        content_hash: u64
```

The index is small. 10K pages × 32 bytes = 320 KB. Loaded entirely
into memory on startup, or mmap'd.

## Loading: single mmap

```rust
use memmap2::Mmap;

struct KVStore {
    config: MappedConfig,
    store: Mmap,              // one mmap for all pages
    index: Vec<IndexEntry>,   // loaded into memory
}

impl KVStore {
    fn open(dir: &Path) -> io::Result<Self> {
        let config = MappedConfig::open(dir.join("config.bin"))?;
        let store_file = File::open(dir.join("store.bin"))?;
        let store = unsafe { Mmap::map(&store_file)? };
        let index = IndexEntry::load_all(dir.join("store.idx"))?;
        Ok(Self { config, store, index })
    }

    fn get_page(&self, slug_hash: u64) -> Option<PageView> {
        let entry = self.index.binary_search_by_key(&slug_hash, |e| e.slug_hash).ok()?;
        let e = &self.index[entry];
        Some(PageView {
            data: &self.store[e.offset as usize .. (e.offset + e.entry_len as u64) as usize],
            config: &self.config,
        })
    }
}

struct PageView<'a> {
    data: &'a [u8],           // slice into the single mmap
    config: &'a MappedConfig,
}

impl<'a> PageView<'a> {
    fn key_quant(&self) -> &[u8] {
        // offset computed from header fields + config params
        let start = 40; // after entry header
        let len = self.num_heads() * self.num_groups() * self.group_size()
                  * (self.config.sketch_dim / 8);
        &self.data[start .. start + len]
    }

    fn key_norms(&self) -> &[f32] {
        let offset = self.key_norms_offset();
        bytemuck::cast_slice(&self.data[offset .. offset + self.key_norms_len()])
    }
}
```

**Cold start:** 1 mmap for config, 1 mmap for store, read index into
memory. Three syscalls total regardless of page count.

**Query hot path:** binary search index → offset → slice into mmap →
XOR + popcount on sign bytes. Sequential scan within each page entry.
OS prefetcher handles the rest.

## Write Path

```
Page change detected (wiki_ingest)
    │
    ▼
1. Tokenize page content
2. Project: K = W_k · tokens, V = W_v · tokens
3. Detect outliers (Algorithm 4)
4. Quantize keys (Algorithm 2)
5. Quantize values (Algorithm 5)
6. Compute norms
7. Serialize entry into buffer
8. Append to store.bin (pwrite at end)
9. fsync store.bin
10. Update index: new entry with generation+1, mark old entry dead
11. Rewrite store.idx atomically (write .tmp, fsync, rename)
```

Only the index rewrite needs atomicity. The store itself is
append-only — a crash mid-append leaves a partial tail entry that
the next open detects (magic check) and truncates.

## Compaction

Dead entries accumulate as pages are updated. Compaction reclaims space.

```
Trigger: dead_bytes > live_bytes * 0.5  (or manual)

Algorithm:
  1. Create store.bin.new
  2. For each live entry in index (sorted by slug_hash):
       Copy entry from store.bin to store.bin.new
       Record new offset
  3. fsync store.bin.new
  4. Build new store.idx.new with updated offsets
  5. fsync store.idx.new
  6. rename store.idx.new → store.idx
  7. rename store.bin.new → store.bin
  8. Readers on old mmap continue until dropped (POSIX semantics)
```

Compaction is offline. Readers are never blocked — they hold the old
mmap until their query completes.

## Staleness Detection

Each entry stores `content_hash` (blake3 of page content, truncated
to u64). On query:

1. Caller provides slug + current content hash
2. Binary search index → compare `content_hash`
3. Match → use compressed state (zero-cost)
4. Mismatch → re-compress and append

Same pattern as tantivy index staleness in llm-wiki.

## Concurrency

| Operation | Lock | Notes |
|-----------|------|-------|
| Read (score query) | None | mmap is read-only, slice into it |
| Append (page update) | Append mutex | pwrite at EOF, no reader impact |
| Index rewrite | Append mutex | atomic rename, readers see old until refresh |
| Compaction | Exclusive | offline, readers on old mmap unaffected |
| Config read | None | immutable after creation |

No locks on the hot path. The append mutex is held only for the
duration of a single pwrite + fsync — milliseconds.

## Rebuild

If the kv-store is deleted or corrupted:

```
llm-wiki kv-store rebuild [--wiki <name>]
```

Re-projects all pages from scratch. Same as `wiki_index_rebuild` for
tantivy — the compressed state is derived, not authoritative.

## Size Estimates

| Wiki size | Pages | Avg tokens/page | store.bin | store.idx |
|-----------|-------|-----------------|-----------|-----------|
| Small | 100 | 500 | ~2 MB | ~3 KB |
| Medium | 1,000 | 500 | ~20 MB | ~32 KB |
| Large | 10,000 | 500 | ~200 MB | ~320 KB |
| Huge | 100,000 | 500 | ~2 GB | ~3.2 MB |

At ~40 bytes/token compressed, 500 tokens/page = ~20 KB/page.

## Crate Dependencies

```toml
memmap2 = "0.9"       # mmap
bytemuck = "1.16"     # zero-copy cast &[u8] → &[f32]
blake3 = "1.5"        # content hashing
```
