use crate::quantize::CompressedKeys;
use crate::store::config::{IndexEntry, IndexMeta, KeysConfig, KEY_ENTRY_MAGIC};
use std::fs::{self, File, OpenOptions};
use std::io::{self, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// Append-only key store with mmap-based reading.
pub struct KeyStore {
    dir: PathBuf,
    pub config: KeysConfig,
    data_mmap: Option<Mmap>,
    index: Vec<IndexEntry>,
    meta: IndexMeta,
    data_len: u64,
    next_generation: u32,
}

/// Zero-copy view into a compressed key entry.
pub struct KeyPageView<'a> {
    data: &'a [u8],
    pub num_vectors: u32,
    pub outlier_count: u8,
    sketch_dim: usize,
    outlier_sketch_dim: usize,
}

impl KeyStore {
    /// Create a new empty key store.
    pub fn create(dir: &Path, config: KeysConfig) -> io::Result<Self> {
        fs::create_dir_all(dir)?;

        // Write empty keys.bin
        File::create(dir.join("keys.bin"))?;

        // Write keys.idx with header + meta + zero entries
        let mut idx_file = File::create(dir.join("keys.idx"))?;
        config.write_to(&mut idx_file)?;
        let meta = IndexMeta::default();
        meta.write_to(&mut idx_file)?;
        idx_file.sync_all()?;

        Ok(Self {
            dir: dir.to_path_buf(),
            config,
            data_mmap: None,
            index: Vec::new(),
            meta,
            data_len: 0,
            next_generation: 1,
        })
    }

    /// Open an existing key store. Recovers from:
    /// - Index entries pointing beyond .bin (dropped)
    /// - Missing index file (rebuilt from .bin scan)
    pub fn open(dir: &Path) -> io::Result<Self> {
        let idx_path = dir.join("keys.idx");
        let data_path = dir.join("keys.bin");

        if !data_path.exists() {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "keys.bin not found",
            ));
        }

        // Truncate any partial tail entry in keys.bin
        let data_len = truncate_partial_tail(&data_path, KEY_ENTRY_MAGIC)?;

        // mmap keys.bin
        let data_file = File::open(&data_path)?;
        let data_mmap = if data_len > 0 {
            Some(unsafe { Mmap::map(&data_file)? })
        } else {
            None
        };

        if idx_path.exists() {
            // Read index, filter out entries beyond data length
            let mut idx_file = File::open(&idx_path)?;
            let config = KeysConfig::read_from(&mut idx_file)?;
            let meta = IndexMeta::read_from(&mut idx_file)?;

            let mut index = Vec::with_capacity(meta.entry_count as usize);
            for _ in 0..meta.entry_count {
                if let Ok(entry) = IndexEntry::read_from(&mut idx_file) {
                    if entry.offset + entry.entry_len as u64 <= data_len {
                        index.push(entry);
                    }
                }
            }

            let live_bytes = index.iter().map(|e| e.entry_len).sum::<u32>();
            let next_generation = index.iter().map(|e| e.generation).max().unwrap_or(0) + 1;

            let store = Self {
                dir: dir.to_path_buf(),
                config,
                data_mmap,
                meta: IndexMeta {
                    entry_count: index.len() as u16,
                    live_bytes,
                    dead_bytes: data_len as u32 - live_bytes,
                },
                index,
                data_len,
                next_generation,
            };

            // Rewrite index if we dropped entries
            if store.meta.entry_count != meta.entry_count {
                store.write_index()?;
            }

            Ok(store)
        } else {
            // No index — rebuild from scanning .bin
            Self::rebuild_from_bin(dir, &data_path, data_len, data_mmap)
        }
    }

    /// Rebuild index by scanning keys.bin entries.
    /// Requires keys.idx to exist (for config header) or a config to be provided.
    fn rebuild_from_bin(
        dir: &Path,
        _data_path: &Path,
        data_len: u64,
        data_mmap: Option<Mmap>,
    ) -> io::Result<Self> {
        // We need the config. Try reading from a backup or require it.
        // For now, scan needs the config — if index is truly gone,
        // the caller must provide config via create() + re-ingest.
        // But if keys.idx.tmp exists (crash during atomic rename), try that.
        let idx_tmp = dir.join("keys.idx.tmp");
        let config = if idx_tmp.exists() {
            let mut f = File::open(&idx_tmp)?;
            KeysConfig::read_from(&mut f)?
        } else {
            return Err(io::Error::new(
                io::ErrorKind::NotFound,
                "keys.idx missing and no backup found; re-create the store",
            ));
        };

        let entries = scan_key_entries(data_mmap.as_deref(), data_len);

        // Deduplicate: keep highest generation per slug_hash
        let mut best: std::collections::HashMap<u64, IndexEntry> = std::collections::HashMap::new();
        for entry in entries {
            best.entry(entry.slug_hash)
                .and_modify(|existing| {
                    if entry.generation > existing.generation {
                        *existing = entry.clone();
                    }
                })
                .or_insert(entry);
        }
        let mut index: Vec<IndexEntry> = best.into_values().collect();
        index.sort_by_key(|e| e.slug_hash);

        let live_bytes = index.iter().map(|e| e.entry_len).sum::<u32>();
        let next_generation = index.iter().map(|e| e.generation).max().unwrap_or(0) + 1;

        let store = Self {
            dir: dir.to_path_buf(),
            config,
            data_mmap,
            meta: IndexMeta {
                entry_count: index.len() as u16,
                live_bytes,
                dead_bytes: data_len as u32 - live_bytes,
            },
            index,
            data_len,
            next_generation,
        };
        store.write_index()?;
        Ok(store)
    }

    /// Append a compressed key entry to the store.
    pub fn append(
        &mut self,
        slug_hash: u64,
        content_hash: u64,
        compressed: &CompressedKeys,
    ) -> io::Result<()> {
        // Serialize entry
        let entry_data = serialize_key_entry(compressed);
        let entry_len = entry_data.len() as u32;

        // Append to keys.bin
        let mut data_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("keys.bin"))?;
        let offset = data_file.seek(SeekFrom::End(0))?;
        data_file.write_all(&entry_data)?;
        data_file.sync_all()?;

        // Track dead space if this slug already exists
        if let Ok(pos) = self.index.binary_search_by_key(&slug_hash, |e| e.slug_hash) {
            self.meta.dead_bytes += self.index[pos].entry_len;
            self.index.remove(pos);
            self.meta.entry_count -= 1;
        }

        // Insert new entry (keep sorted)
        let generation = self.next_generation;
        self.next_generation += 1;
        let new_entry = IndexEntry {
            slug_hash,
            offset,
            entry_len,
            generation,
            content_hash,
        };
        let insert_pos = self
            .index
            .binary_search_by_key(&slug_hash, |e| e.slug_hash)
            .unwrap_err();
        self.index.insert(insert_pos, new_entry);
        self.meta.entry_count += 1;
        self.meta.live_bytes += entry_len;
        self.data_len = offset + entry_len as u64;

        // Rewrite index atomically
        self.write_index()?;

        // Re-mmap
        let data_file = File::open(self.dir.join("keys.bin"))?;
        self.data_mmap = Some(unsafe { Mmap::map(&data_file)? });

        Ok(())
    }

    /// Look up a page by slug hash. Returns a zero-copy view.
    pub fn get_page(&self, slug_hash: u64) -> Option<KeyPageView<'_>> {
        let pos = self
            .index
            .binary_search_by_key(&slug_hash, |e| e.slug_hash)
            .ok()?;
        let entry = &self.index[pos];
        let mmap = self.data_mmap.as_ref()?;
        let start = entry.offset as usize;
        let end = start + entry.entry_len as usize;
        if end > mmap.len() {
            return None;
        }
        KeyPageView::parse(
            &mmap[start..end],
            self.config.sketch_dim as usize,
            self.config.outlier_sketch_dim as usize,
        )
    }

    /// Check if a page's compressed keys are fresh.
    pub fn is_fresh(&self, slug_hash: u64, content_hash: u64) -> bool {
        self.index
            .binary_search_by_key(&slug_hash, |e| e.slug_hash)
            .ok()
            .map(|i| self.index[i].content_hash == content_hash)
            .unwrap_or(false)
    }

    /// Number of pages in the store.
    pub fn len(&self) -> usize {
        self.index.len()
    }

    /// Whether the store is empty.
    pub fn is_empty(&self) -> bool {
        self.index.is_empty()
    }

    pub fn live_bytes(&self) -> u32 {
        self.meta.live_bytes
    }

    pub fn dead_bytes(&self) -> u32 {
        self.meta.dead_bytes
    }

    /// Compact the store — rewrite only live entries, reclaim dead space.
    pub fn compact(&mut self) -> io::Result<()> {
        let data_path = self.dir.join("keys.bin");
        let tmp_path = self.dir.join("keys.bin.compact");

        // Read current data
        let old_mmap = self.data_mmap.take();
        let old_data = match &old_mmap {
            Some(m) => &m[..],
            None => return Ok(()), // nothing to compact
        };

        // Write live entries to new file
        let mut new_file = File::create(&tmp_path)?;
        let mut new_index = Vec::with_capacity(self.index.len());

        for entry in &self.index {
            let start = entry.offset as usize;
            let end = start + entry.entry_len as usize;
            if end > old_data.len() {
                continue;
            }
            let new_offset = new_file.seek(SeekFrom::End(0))?;
            new_file.write_all(&old_data[start..end])?;
            new_index.push(IndexEntry {
                offset: new_offset,
                ..entry.clone()
            });
        }
        new_file.sync_all()?;

        // Atomic rename
        fs::rename(&tmp_path, &data_path)?;

        // Update state
        self.meta.dead_bytes = 0;
        self.meta.live_bytes = new_index.iter().map(|e| e.entry_len).sum();
        self.meta.entry_count = new_index.len() as u16;
        self.index = new_index;
        self.write_index()?;

        // Re-mmap
        let data_file = File::open(&data_path)?;
        self.data_len = data_file.metadata()?.len();
        self.data_mmap = if self.data_len > 0 {
            Some(unsafe { Mmap::map(&data_file)? })
        } else {
            None
        };

        Ok(())
    }

    fn write_index(&self) -> io::Result<()> {
        let tmp_path = self.dir.join("keys.idx.tmp");
        let final_path = self.dir.join("keys.idx");

        let mut f = File::create(&tmp_path)?;
        self.config.write_to(&mut f)?;
        self.meta.write_to(&mut f)?;
        for entry in &self.index {
            entry.write_to(&mut f)?;
        }
        f.sync_all()?;
        fs::rename(tmp_path, final_path)?;
        Ok(())
    }
}

// ── Entry serialization ───────────────────────────────────────────────────────

/// Truncate a .bin file to the last valid entry.
/// Walks entries by magic + entry_len. If the tail is partial, truncates.
/// Returns the valid data length.
fn truncate_partial_tail(path: &Path, magic: &[u8; 4]) -> io::Result<u64> {
    let mut file = OpenOptions::new().read(true).write(true).open(path)?;
    let file_len = file.metadata()?.len();
    if file_len == 0 {
        return Ok(0);
    }

    let mut data = vec![0u8; file_len as usize];
    file.seek(SeekFrom::Start(0))?;
    use std::io::Read;
    file.read_exact(&mut data)?;

    let mut offset = 0usize;
    let mut last_good = 0usize;

    while offset + 8 <= data.len() {
        if &data[offset..offset + 4] != magic {
            break;
        }
        let entry_len =
            u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if entry_len < 8 || offset + entry_len > data.len() {
            break;
        }
        last_good = offset + entry_len;
        offset = last_good;
    }

    if last_good < data.len() {
        file.set_len(last_good as u64)?;
        file.sync_all()?;
    }

    Ok(last_good as u64)
}

/// Scan keys.bin and extract index entries from entry headers.
fn scan_key_entries(data: Option<&[u8]>, data_len: u64) -> Vec<IndexEntry> {
    let data = match data {
        Some(d) => d,
        None => return Vec::new(),
    };

    let mut entries = Vec::new();
    let mut offset = 0usize;
    let mut generation = 1u32;

    while offset + 8 <= data.len() {
        if &data[offset..offset + 4] != KEY_ENTRY_MAGIC {
            break;
        }
        let entry_len =
            u32::from_le_bytes(data[offset + 4..offset + 8].try_into().unwrap()) as usize;
        if entry_len < 16 || offset + entry_len > data_len as usize {
            break;
        }

        entries.push(IndexEntry {
            slug_hash: 0,
            offset: offset as u64,
            entry_len: entry_len as u32,
            generation,
            content_hash: 0,
        });

        generation += 1;
        offset += entry_len;
    }

    entries
}

fn serialize_key_entry(compressed: &CompressedKeys) -> Vec<u8> {
    let outlier_count = compressed.outlier_indices.len() as u8;

    let header_size = 4 + 4 + 4 + 1 + 3; // magic + entry_len + num_vectors + outlier_count + padding
    let body_size = compressed.outlier_indices.len()
        + compressed.key_quant.len()
        + compressed.key_outlier_quant.len()
        + compressed.key_norms.len() * 4
        + compressed.outlier_norms.len() * 4;
    let entry_len = header_size + body_size;

    let mut buf = Vec::with_capacity(entry_len);
    buf.extend_from_slice(KEY_ENTRY_MAGIC);
    buf.extend_from_slice(&(entry_len as u32).to_le_bytes());
    buf.extend_from_slice(&(compressed.num_vectors as u32).to_le_bytes());
    buf.push(outlier_count);
    buf.extend_from_slice(&[0u8; 3]); // padding

    buf.extend_from_slice(&compressed.outlier_indices);
    buf.extend_from_slice(&compressed.key_quant);
    buf.extend_from_slice(&compressed.key_outlier_quant);
    for &n in &compressed.key_norms {
        buf.extend_from_slice(&n.to_le_bytes());
    }
    for &n in &compressed.outlier_norms {
        buf.extend_from_slice(&n.to_le_bytes());
    }

    buf
}

// ── KeyPageView ───────────────────────────────────────────────────────────────

impl<'a> KeyPageView<'a> {
    fn parse(data: &'a [u8], sketch_dim: usize, outlier_sketch_dim: usize) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }
        if &data[0..4] != KEY_ENTRY_MAGIC {
            return None;
        }
        let num_vectors = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let outlier_count = data[12];

        Some(Self {
            data,
            num_vectors,
            outlier_count,
            sketch_dim,
            outlier_sketch_dim,
        })
    }

    const HEADER_SIZE: usize = 16; // magic(4) + entry_len(4) + num_vectors(4) + outlier_count(1) + padding(3)

    fn outlier_indices_offset(&self) -> usize {
        Self::HEADER_SIZE
    }

    fn key_quant_offset(&self) -> usize {
        self.outlier_indices_offset() + self.outlier_count as usize
    }

    fn key_quant_len(&self) -> usize {
        self.num_vectors as usize * (self.sketch_dim / 8)
    }

    fn key_outlier_quant_offset(&self) -> usize {
        self.key_quant_offset() + self.key_quant_len()
    }

    fn key_outlier_quant_len(&self) -> usize {
        self.num_vectors as usize * (self.outlier_sketch_dim / 8)
    }

    fn key_norms_offset(&self) -> usize {
        self.key_outlier_quant_offset() + self.key_outlier_quant_len()
    }

    fn key_norms_len(&self) -> usize {
        self.num_vectors as usize * 4
    }

    fn outlier_norms_offset(&self) -> usize {
        self.key_norms_offset() + self.key_norms_len()
    }

    pub fn outlier_indices(&self) -> &[u8] {
        let start = self.outlier_indices_offset();
        &self.data[start..start + self.outlier_count as usize]
    }

    pub fn key_quant(&self) -> &[u8] {
        let start = self.key_quant_offset();
        &self.data[start..start + self.key_quant_len()]
    }

    pub fn key_outlier_quant(&self) -> &[u8] {
        let start = self.key_outlier_quant_offset();
        &self.data[start..start + self.key_outlier_quant_len()]
    }

    pub fn key_norms(&self) -> Vec<f32> {
        let start = self.key_norms_offset();
        let count = self.num_vectors as usize;
        read_f32_slice(&self.data[start..start + count * 4], count)
    }

    pub fn outlier_norms(&self) -> Vec<f32> {
        let start = self.outlier_norms_offset();
        let count = self.num_vectors as usize;
        read_f32_slice(&self.data[start..start + count * 4], count)
    }

    /// Reconstruct a `CompressedKeys` from the view (copies data).
    pub fn to_compressed_keys(&self, head_dim: usize) -> CompressedKeys {
        CompressedKeys {
            key_quant: self.key_quant().to_vec(),
            key_outlier_quant: self.key_outlier_quant().to_vec(),
            key_norms: self.key_norms(),
            outlier_norms: self.outlier_norms(),
            outlier_indices: self.outlier_indices().to_vec(),
            num_vectors: self.num_vectors as usize,
            head_dim,
        }
    }
}

/// Read f32 values from a byte slice (handles unaligned data).
fn read_f32_slice(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * 4;
        let val = f32::from_le_bytes(bytes[start..start + 4].try_into().unwrap());
        out.push(val);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
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

    fn test_config() -> KeysConfig {
        KeysConfig {
            head_dim: 16,
            sketch_dim: 32,
            outlier_sketch_dim: 16,
            seed: 42,
        }
    }

    #[test]
    fn test_create_and_open_empty() {
        let dir = tempdir().unwrap();
        let config = test_config();
        KeyStore::create(dir.path(), config.clone()).unwrap();

        let store = KeyStore::open(dir.path()).unwrap();
        assert_eq!(store.config, config);
        assert!(store.is_empty());
        assert_eq!(store.len(), 0);
    }

    #[test]
    fn test_append_and_get() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let keys = random_vec(4 * 16, &mut rng);
        let outlier_indices = vec![0u8, 1];
        let compressed = sketch.quantize(&keys, 4, &outlier_indices);

        let slug_hash = 0xAABB;
        let content_hash = 0xCCDD;
        store.append(slug_hash, content_hash, &compressed).unwrap();

        assert_eq!(store.len(), 1);
        let page = store.get_page(slug_hash).unwrap();
        assert_eq!(page.num_vectors, 4);
        assert_eq!(page.key_quant(), compressed.key_quant.as_slice());
        assert_eq!(page.key_norms(), compressed.key_norms.as_slice());
        assert_eq!(page.outlier_norms(), compressed.outlier_norms.as_slice());
        assert_eq!(
            page.outlier_indices(),
            compressed.outlier_indices.as_slice()
        );
    }

    #[test]
    fn test_page_not_found() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let store = KeyStore::create(dir.path(), config).unwrap();
        assert!(store.get_page(0xDEAD).is_none());
    }

    #[test]
    fn test_score_survives_persistence() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(456);
        let keys = random_vec(8 * 16, &mut rng);
        let query = random_vec(16, &mut rng);
        let outlier_indices = vec![0u8];
        let compressed = sketch.quantize(&keys, 8, &outlier_indices);

        let scores_before = sketch.score(&query, &compressed);

        store.append(0x1234, 0x5678, &compressed).unwrap();

        // Reopen from disk
        let store2 = KeyStore::open(dir.path()).unwrap();
        let sketch2 = store2.config.build_sketch();
        let page = store2.get_page(0x1234).unwrap();
        let reloaded = page.to_compressed_keys(config.head_dim as usize);
        let scores_after = sketch2.score(&query, &reloaded);

        assert_eq!(scores_before, scores_after);
    }

    #[test]
    fn test_multiple_pages() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(789);
        for slug in 0u64..5 {
            let keys = random_vec(4 * 16, &mut rng);
            let compressed = sketch.quantize(&keys, 4, &[0u8]);
            store.append(slug, slug * 100, &compressed).unwrap();
        }

        assert_eq!(store.len(), 5);
        for slug in 0u64..5 {
            assert!(store.get_page(slug).is_some());
        }
        assert!(store.get_page(99).is_none());
    }

    #[test]
    fn test_reopen_preserves_all_pages() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(111);
        for slug in 0u64..3 {
            let keys = random_vec(4 * 16, &mut rng);
            let compressed = sketch.quantize(&keys, 4, &[0u8]);
            store.append(slug, slug, &compressed).unwrap();
        }

        let store2 = KeyStore::open(dir.path()).unwrap();
        assert_eq!(store2.len(), 3);
        for slug in 0u64..3 {
            assert!(store2.get_page(slug).is_some());
        }
    }

    #[test]
    fn test_staleness() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(222);
        let keys = random_vec(4 * 16, &mut rng);
        let compressed = sketch.quantize(&keys, 4, &[0u8]);
        store.append(0xAA, 0xBB, &compressed).unwrap();

        assert!(store.is_fresh(0xAA, 0xBB));
        assert!(!store.is_fresh(0xAA, 0xCC));
        assert!(!store.is_fresh(0xFF, 0xBB));
    }

    #[test]
    fn test_update_overwrites_old() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = KeyStore::create(dir.path(), config.clone()).unwrap();
        let sketch = config.build_sketch();

        let mut rng = ChaCha20Rng::seed_from_u64(333);
        let keys_v1 = random_vec(4 * 16, &mut rng);
        let compressed_v1 = sketch.quantize(&keys_v1, 4, &[0u8]);
        store.append(0xAA, 0x11, &compressed_v1).unwrap();

        let keys_v2 = random_vec(4 * 16, &mut rng);
        let compressed_v2 = sketch.quantize(&keys_v2, 4, &[0u8]);
        store.append(0xAA, 0x22, &compressed_v2).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.is_fresh(0xAA, 0x22));
        assert!(!store.is_fresh(0xAA, 0x11));
        assert!(store.dead_bytes() > 0);
    }
}
