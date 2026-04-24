// GPU acceleration via WGPU compute shaders.
//
// GPU is used only by `KeyStore::score_all_pages` which batches all
// vectors across all pages into a single GPU dispatch. Individual
// `score()` and `score_compressed()` calls always use CPU — single
// pages (32 vectors) are too small for GPU to beat CPU.
//
// Override threshold: `QJL_GPU_MIN_BATCH` env var (default 5000).

#[cfg(feature = "gpu")]
mod wgpu_backend;

#[cfg(feature = "gpu")]
pub use wgpu_backend::GpuContext;

#[cfg(feature = "gpu")]
const DEFAULT_GPU_MIN_BATCH: usize = 5_000;

/// Minimum total vectors across all pages for GPU dispatch in `score_all_pages`.
///
/// Override: `QJL_GPU_MIN_BATCH=0` to always use GPU.
#[cfg(feature = "gpu")]
pub fn gpu_min_batch() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("QJL_GPU_MIN_BATCH")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GPU_MIN_BATCH)
    })
}
