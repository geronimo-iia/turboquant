use crate::error::{QjlError, Result};
use crate::store::config::{IndexEntry, IndexMeta, ValuesConfig, VALUE_ENTRY_MAGIC};
use crate::values::CompressedValues;
use std::fs::{self, File, OpenOptions};
use std::io::{Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};

use memmap2::Mmap;

/// Append-only value store with mmap-based reading.
pub struct ValueStore {
    dir: PathBuf,
    pub config: ValuesConfig,
    data_mmap: Option<Mmap>,
    index: Vec<IndexEntry>,
    meta: IndexMeta,
    data_len: u64,
    next_generation: u32,
}

/// Zero-copy view into a compressed value entry.
pub struct ValuePageView<'a> {
    data: &'a [u8],
    pub num_elements: u32,
    pub num_groups: u32,
    bits: u8,
    group_size: usize,
}

impl ValueStore {
    /// Create a new empty value store.
    pub fn create(dir: &Path, config: ValuesConfig) -> Result<Self> {
        fs::create_dir_all(dir)?;

        File::create(dir.join("values.bin"))?;

        let mut idx_file = File::create(dir.join("values.idx"))?;
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

    /// Open an existing value store. Recovers from:
    /// - Truncated tail in values.bin (partial entry removed)
    /// - Index entries pointing beyond values.bin (dropped)
    pub fn open(dir: &Path) -> Result<Self> {
        let idx_path = dir.join("values.idx");
        let data_path = dir.join("values.bin");

        if !data_path.exists() {
            return Err(QjlError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "values.bin not found",
            )));
        }

        // Truncate any partial tail entry
        let data_len = truncate_partial_tail(&data_path, VALUE_ENTRY_MAGIC)?;

        let data_file = File::open(&data_path)?;
        let data_mmap = if data_len > 0 {
            Some(unsafe { Mmap::map(&data_file)? })
        } else {
            None
        };

        if !idx_path.exists() {
            return Err(QjlError::Io(std::io::Error::new(
                std::io::ErrorKind::NotFound,
                "values.idx not found",
            )));
        }

        let mut idx_file = File::open(&idx_path)?;
        let config = ValuesConfig::read_from(&mut idx_file)?;
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

        if store.meta.entry_count != meta.entry_count {
            store.write_index()?;
        }

        Ok(store)
    }

    /// Append a compressed value entry to the store.
    pub fn append(
        &mut self,
        slug_hash: u64,
        content_hash: u64,
        compressed: &CompressedValues,
    ) -> Result<()> {
        let entry_data = serialize_value_entry(compressed);
        let entry_len = entry_data.len() as u32;

        let mut data_file = OpenOptions::new()
            .create(true)
            .append(true)
            .open(self.dir.join("values.bin"))?;
        let offset = data_file.seek(SeekFrom::End(0))?;
        data_file.write_all(&entry_data)?;
        data_file.sync_all()?;

        if let Ok(pos) = self.index.binary_search_by_key(&slug_hash, |e| e.slug_hash) {
            self.meta.dead_bytes += self.index[pos].entry_len;
            self.index.remove(pos);
            self.meta.entry_count -= 1;
        }

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

        self.write_index()?;

        let data_file = File::open(self.dir.join("values.bin"))?;
        self.data_mmap = Some(unsafe { Mmap::map(&data_file)? });

        Ok(())
    }

    /// Look up a page by slug hash.
    pub fn get_page(&self, slug_hash: u64) -> Option<ValuePageView<'_>> {
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
        ValuePageView::parse(
            &mmap[start..end],
            self.config.bits,
            self.config.group_size as usize,
        )
    }

    /// Check if a page's compressed values are fresh.
    pub fn is_fresh(&self, slug_hash: u64, content_hash: u64) -> bool {
        self.index
            .binary_search_by_key(&slug_hash, |e| e.slug_hash)
            .ok()
            .map(|i| self.index[i].content_hash == content_hash)
            .unwrap_or(false)
    }

    pub fn len(&self) -> usize {
        self.index.len()
    }

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
    pub fn compact(&mut self) -> Result<()> {
        let data_path = self.dir.join("values.bin");
        let tmp_path = self.dir.join("values.bin.compact");

        let old_mmap = self.data_mmap.take();
        let old_data = match &old_mmap {
            Some(m) => &m[..],
            None => return Ok(()),
        };

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

        fs::rename(&tmp_path, &data_path)?;

        self.meta.dead_bytes = 0;
        self.meta.live_bytes = new_index.iter().map(|e| e.entry_len).sum();
        self.meta.entry_count = new_index.len() as u16;
        self.index = new_index;
        self.write_index()?;

        let data_file = File::open(&data_path)?;
        self.data_len = data_file.metadata()?.len();
        self.data_mmap = if self.data_len > 0 {
            Some(unsafe { Mmap::map(&data_file)? })
        } else {
            None
        };

        Ok(())
    }

    fn write_index(&self) -> Result<()> {
        let tmp_path = self.dir.join("values.idx.tmp");
        let final_path = self.dir.join("values.idx");

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
fn truncate_partial_tail(path: &Path, magic: &[u8; 4]) -> Result<u64> {
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

fn serialize_value_entry(compressed: &CompressedValues) -> Vec<u8> {
    let num_groups = compressed.scale.len();

    // header: magic(4) + entry_len(4) + num_elements(4) + num_groups(4) = 16
    let header_size = 16;
    let body_size = compressed.packed.len() * 4 + num_groups * 4 + num_groups * 4;
    let entry_len = header_size + body_size;

    let mut buf = Vec::with_capacity(entry_len);
    buf.extend_from_slice(VALUE_ENTRY_MAGIC);
    buf.extend_from_slice(&(entry_len as u32).to_le_bytes());
    buf.extend_from_slice(&(compressed.num_elements as u32).to_le_bytes());
    buf.extend_from_slice(&(num_groups as u32).to_le_bytes());

    for &v in &compressed.packed {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &compressed.scale {
        buf.extend_from_slice(&v.to_le_bytes());
    }
    for &v in &compressed.mn {
        buf.extend_from_slice(&v.to_le_bytes());
    }

    buf
}

// ── ValuePageView ─────────────────────────────────────────────────────────────

impl<'a> ValuePageView<'a> {
    fn parse(data: &'a [u8], bits: u8, group_size: usize) -> Option<Self> {
        if data.len() < 16 {
            return None;
        }
        if &data[0..4] != VALUE_ENTRY_MAGIC {
            return None;
        }
        let num_elements = u32::from_le_bytes(data[8..12].try_into().ok()?);
        let num_groups = u32::from_le_bytes(data[12..16].try_into().ok()?);

        Some(Self {
            data,
            num_elements,
            num_groups,
            bits,
            group_size,
        })
    }

    const HEADER_SIZE: usize = 16;

    fn packed_offset(&self) -> usize {
        Self::HEADER_SIZE
    }

    fn packed_len(&self) -> usize {
        let feat_per_int = 32 / self.bits as usize;
        (self.num_elements as usize / feat_per_int) * 4
    }

    fn scale_offset(&self) -> usize {
        self.packed_offset() + self.packed_len()
    }

    fn scale_len(&self) -> usize {
        self.num_groups as usize * 4
    }

    fn mn_offset(&self) -> usize {
        self.scale_offset() + self.scale_len()
    }

    pub fn packed(&self) -> Vec<i32> {
        let start = self.packed_offset();
        let count = self.packed_len() / 4;
        read_i32_slice(&self.data[start..start + count * 4], count)
    }

    pub fn scale(&self) -> Vec<f32> {
        let start = self.scale_offset();
        let count = self.num_groups as usize;
        read_f32_slice(&self.data[start..start + count * 4], count)
    }

    pub fn mn(&self) -> Vec<f32> {
        let start = self.mn_offset();
        let count = self.num_groups as usize;
        read_f32_slice(&self.data[start..start + count * 4], count)
    }

    /// Reconstruct a `CompressedValues` from the view (copies data).
    pub fn to_compressed_values(&self) -> CompressedValues {
        CompressedValues {
            packed: self.packed(),
            scale: self.scale(),
            mn: self.mn(),
            num_elements: self.num_elements as usize,
            bits: self.bits,
            group_size: self.group_size,
        }
    }
}

fn read_f32_slice(bytes: &[u8], count: usize) -> Vec<f32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * 4;
        out.push(f32::from_le_bytes(
            bytes[start..start + 4].try_into().unwrap(),
        ));
    }
    out
}

fn read_i32_slice(bytes: &[u8], count: usize) -> Vec<i32> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        let start = i * 4;
        out.push(i32::from_le_bytes(
            bytes[start..start + 4].try_into().unwrap(),
        ));
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::values::{quantize_values, quantized_dot};
    use tempfile::tempdir;

    fn test_config() -> ValuesConfig {
        ValuesConfig {
            bits: 4,
            group_size: 8,
        }
    }

    #[test]
    fn test_create_and_open_empty() {
        let dir = tempdir().unwrap();
        let config = test_config();
        ValueStore::create(dir.path(), config.clone()).unwrap();

        let store = ValueStore::open(dir.path()).unwrap();
        assert_eq!(store.config, config);
        assert!(store.is_empty());
    }

    #[test]
    fn test_append_and_get() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = ValueStore::create(dir.path(), config).unwrap();

        let values: Vec<f32> = (0..16).map(|i| i as f32 * 0.5).collect();
        let compressed = quantize_values(&values, 8, 4).unwrap();

        store.append(0xAA, 0xBB, &compressed).unwrap();

        assert_eq!(store.len(), 1);
        let page = store.get_page(0xAA).unwrap();
        assert_eq!(page.num_elements, compressed.num_elements as u32);
        assert_eq!(page.packed(), compressed.packed);
        assert_eq!(page.scale(), compressed.scale);
        assert_eq!(page.mn(), compressed.mn);
    }

    #[test]
    fn test_page_not_found() {
        let dir = tempdir().unwrap();
        let store = ValueStore::create(dir.path(), test_config()).unwrap();
        assert!(store.get_page(0xDEAD).is_none());
    }

    #[test]
    fn test_quantized_dot_survives_persistence() {
        let dir = tempdir().unwrap();
        let config = test_config();
        let mut store = ValueStore::create(dir.path(), config).unwrap();

        let values: Vec<f32> = (0..16).map(|i| i as f32).collect();
        let weights: Vec<f32> = (0..16).map(|i| (i as f32) * 0.1).collect();
        let compressed = quantize_values(&values, 8, 4).unwrap();

        let dot_before = quantized_dot(&weights, &compressed).unwrap();

        store.append(0x11, 0x22, &compressed).unwrap();

        let store2 = ValueStore::open(dir.path()).unwrap();
        let page = store2.get_page(0x11).unwrap();
        let reloaded = page.to_compressed_values();
        let dot_after = quantized_dot(&weights, &reloaded).unwrap();

        assert_eq!(dot_before, dot_after);
    }

    #[test]
    fn test_multiple_pages() {
        let dir = tempdir().unwrap();
        let mut store = ValueStore::create(dir.path(), test_config()).unwrap();

        for slug in 0u64..5 {
            let values: Vec<f32> = (0..8).map(|i| (slug as f32) + i as f32).collect();
            let compressed = quantize_values(&values, 8, 4).unwrap();
            store.append(slug, slug * 10, &compressed).unwrap();
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
        let mut store = ValueStore::create(dir.path(), test_config()).unwrap();

        for slug in 0u64..3 {
            let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
            let compressed = quantize_values(&values, 8, 4).unwrap();
            store.append(slug, slug, &compressed).unwrap();
        }

        let store2 = ValueStore::open(dir.path()).unwrap();
        assert_eq!(store2.len(), 3);
        for slug in 0u64..3 {
            assert!(store2.get_page(slug).is_some());
        }
    }

    #[test]
    fn test_staleness() {
        let dir = tempdir().unwrap();
        let mut store = ValueStore::create(dir.path(), test_config()).unwrap();

        let values: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let compressed = quantize_values(&values, 8, 4).unwrap();
        store.append(0xAA, 0xBB, &compressed).unwrap();

        assert!(store.is_fresh(0xAA, 0xBB));
        assert!(!store.is_fresh(0xAA, 0xCC));
        assert!(!store.is_fresh(0xFF, 0xBB));
    }

    #[test]
    fn test_update_overwrites_old() {
        let dir = tempdir().unwrap();
        let mut store = ValueStore::create(dir.path(), test_config()).unwrap();

        let v1: Vec<f32> = (0..8).map(|i| i as f32).collect();
        let c1 = quantize_values(&v1, 8, 4).unwrap();
        store.append(0xAA, 0x11, &c1).unwrap();

        let v2: Vec<f32> = (0..8).map(|i| i as f32 * 2.0).collect();
        let c2 = quantize_values(&v2, 8, 4).unwrap();
        store.append(0xAA, 0x22, &c2).unwrap();

        assert_eq!(store.len(), 1);
        assert!(store.is_fresh(0xAA, 0x22));
        assert!(!store.is_fresh(0xAA, 0x11));
        assert!(store.dead_bytes() > 0);
    }
}
