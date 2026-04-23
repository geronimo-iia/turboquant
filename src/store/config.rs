use crate::error::{QjlError, Result};
use crate::sketch::QJLSketch;
use std::io::{Read, Write};

pub const KEYS_INDEX_MAGIC: &[u8; 4] = b"TQKI";
pub const VALUES_INDEX_MAGIC: &[u8; 4] = b"TQVI";
pub const KEY_ENTRY_MAGIC: &[u8; 4] = b"TQKE";
pub const VALUE_ENTRY_MAGIC: &[u8; 4] = b"TQVE";

pub const INDEX_VERSION: u16 = 1;

#[derive(Clone, Debug, PartialEq)]
pub struct KeysConfig {
    pub head_dim: u16,
    pub sketch_dim: u16,
    pub outlier_sketch_dim: u16,
    pub seed: u64,
}

impl KeysConfig {
    pub fn write_to(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(KEYS_INDEX_MAGIC)?;
        w.write_all(&INDEX_VERSION.to_le_bytes())?;
        w.write_all(&self.head_dim.to_le_bytes())?;
        w.write_all(&self.sketch_dim.to_le_bytes())?;
        w.write_all(&self.outlier_sketch_dim.to_le_bytes())?;
        w.write_all(&self.seed.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != KEYS_INDEX_MAGIC {
            return Err(QjlError::StoreMagicMismatch);
        }
        let version = read_u16(r)?;
        if version != INDEX_VERSION {
            return Err(QjlError::StoreVersionMismatch {
                expected: INDEX_VERSION,
                got: version,
            });
        }
        Ok(Self {
            head_dim: read_u16(r)?,
            sketch_dim: read_u16(r)?,
            outlier_sketch_dim: read_u16(r)?,
            seed: read_u64(r)?,
        })
    }

    pub fn build_sketch(&self) -> QJLSketch {
        QJLSketch::new(
            self.head_dim as usize,
            self.sketch_dim as usize,
            self.outlier_sketch_dim as usize,
            self.seed,
        )
        .expect("KeysConfig contains invalid sketch params")
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ValuesConfig {
    pub bits: u8,
    pub group_size: u16,
}

impl ValuesConfig {
    pub fn write_to(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(VALUES_INDEX_MAGIC)?;
        w.write_all(&INDEX_VERSION.to_le_bytes())?;
        w.write_all(&[self.bits, 0])?;
        w.write_all(&self.group_size.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> Result<Self> {
        let mut magic = [0u8; 4];
        r.read_exact(&mut magic)?;
        if &magic != VALUES_INDEX_MAGIC {
            return Err(QjlError::StoreMagicMismatch);
        }
        let version = read_u16(r)?;
        if version != INDEX_VERSION {
            return Err(QjlError::StoreVersionMismatch {
                expected: INDEX_VERSION,
                got: version,
            });
        }
        let bits = read_u8(r)?;
        let _padding = read_u8(r)?;
        let group_size = read_u16(r)?;
        Ok(Self { bits, group_size })
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct IndexEntry {
    pub slug_hash: u64,
    pub offset: u64,
    pub entry_len: u32,
    pub generation: u32,
    pub content_hash: u64,
}

impl IndexEntry {
    pub const SIZE: usize = 32;

    pub fn write_to(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(&self.slug_hash.to_le_bytes())?;
        w.write_all(&self.offset.to_le_bytes())?;
        w.write_all(&self.entry_len.to_le_bytes())?;
        w.write_all(&self.generation.to_le_bytes())?;
        w.write_all(&self.content_hash.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> Result<Self> {
        Ok(Self {
            slug_hash: read_u64(r)?,
            offset: read_u64(r)?,
            entry_len: read_u32(r)?,
            generation: read_u32(r)?,
            content_hash: read_u64(r)?,
        })
    }
}

#[derive(Clone, Debug, Default)]
pub struct IndexMeta {
    pub entry_count: u16,
    pub live_bytes: u32,
    pub dead_bytes: u32,
}

impl IndexMeta {
    pub fn write_to(&self, w: &mut impl Write) -> Result<()> {
        w.write_all(&self.entry_count.to_le_bytes())?;
        w.write_all(&[0u8; 2])?;
        w.write_all(&self.live_bytes.to_le_bytes())?;
        w.write_all(&self.dead_bytes.to_le_bytes())?;
        Ok(())
    }

    pub fn read_from(r: &mut impl Read) -> Result<Self> {
        let entry_count = read_u16(r)?;
        let _padding = read_u16(r)?;
        let live_bytes = read_u32(r)?;
        let dead_bytes = read_u32(r)?;
        Ok(Self {
            entry_count,
            live_bytes,
            dead_bytes,
        })
    }
}

fn read_u8(r: &mut impl Read) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf)?;
    Ok(buf[0])
}

fn read_u16(r: &mut impl Read) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32(r: &mut impl Read) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64(r: &mut impl Read) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf)?;
    Ok(u64::from_le_bytes(buf))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_keys_config_round_trip() {
        let config = KeysConfig {
            head_dim: 128,
            sketch_dim: 256,
            outlier_sketch_dim: 64,
            seed: 42,
        };
        let mut buf = Vec::new();
        config.write_to(&mut buf).unwrap();
        let loaded = KeysConfig::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(config, loaded);
    }

    #[test]
    fn test_values_config_round_trip() {
        let config = ValuesConfig {
            bits: 4,
            group_size: 32,
        };
        let mut buf = Vec::new();
        config.write_to(&mut buf).unwrap();
        let loaded = ValuesConfig::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(config, loaded);
    }

    #[test]
    fn test_index_entry_round_trip() {
        let entry = IndexEntry {
            slug_hash: 0xDEADBEEF,
            offset: 1024,
            entry_len: 500,
            generation: 3,
            content_hash: 0xCAFEBABE,
        };
        let mut buf = Vec::new();
        entry.write_to(&mut buf).unwrap();
        assert_eq!(buf.len(), IndexEntry::SIZE);
        let loaded = IndexEntry::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(entry, loaded);
    }

    #[test]
    fn test_index_meta_round_trip() {
        let meta = IndexMeta {
            entry_count: 42,
            live_bytes: 10000,
            dead_bytes: 500,
        };
        let mut buf = Vec::new();
        meta.write_to(&mut buf).unwrap();
        let loaded = IndexMeta::read_from(&mut Cursor::new(&buf)).unwrap();
        assert_eq!(meta.entry_count, loaded.entry_count);
        assert_eq!(meta.live_bytes, loaded.live_bytes);
        assert_eq!(meta.dead_bytes, loaded.dead_bytes);
    }

    #[test]
    fn test_sketch_reconstruction() {
        let config = KeysConfig {
            head_dim: 64,
            sketch_dim: 128,
            outlier_sketch_dim: 32,
            seed: 42,
        };
        let sketch_a = config.build_sketch();
        let sketch_b = config.build_sketch();
        assert_eq!(sketch_a.proj_dir_score, sketch_b.proj_dir_score);
        assert_eq!(sketch_a.head_dim, 64);
        assert_eq!(sketch_a.sketch_dim, 128);
    }

    #[test]
    fn test_bad_magic_rejected() {
        let buf = b"BADMxxxxxx";
        let result = KeysConfig::read_from(&mut Cursor::new(buf));
        assert!(matches!(result, Err(QjlError::StoreMagicMismatch)));
    }
}
