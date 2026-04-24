// GPU acceleration via WGPU compute shaders.
//
// Provides transparent runtime dispatch: if a GPU adapter is available
// and the batch is large enough, scoring runs on GPU. Otherwise falls
// back to CPU silently.
//
// Two separate thresholds:
// - `QJL_GPU_MIN_BATCH` — for float×sign `score()` (default 5000)
// - `QJL_GPU_MIN_BATCH_COMPRESSED` — for compressed `score_compressed()` (default 100000)
//
// Compressed scoring is ~17 ns/vector on CPU (byte XOR + popcount),
// so GPU only wins at very large batches. Float×sign is ~0.6 µs/vector,
// making GPU worthwhile much sooner.

#[cfg(feature = "gpu")]
mod wgpu_backend;

#[cfg(feature = "gpu")]
pub use wgpu_backend::GpuContext;

#[cfg(feature = "gpu")]
const DEFAULT_GPU_MIN_BATCH: usize = 5000;

#[cfg(feature = "gpu")]
const DEFAULT_GPU_MIN_BATCH_COMPRESSED: usize = 100_000;

/// Minimum vectors for float×sign `score()` GPU dispatch.
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

/// Minimum vectors for compressed `score_compressed()` GPU dispatch.
///
/// Override: `QJL_GPU_MIN_BATCH_COMPRESSED=0` to always use GPU.
#[cfg(feature = "gpu")]
pub fn gpu_min_batch_compressed() -> usize {
    use std::sync::OnceLock;
    static CACHED: OnceLock<usize> = OnceLock::new();
    *CACHED.get_or_init(|| {
        std::env::var("QJL_GPU_MIN_BATCH_COMPRESSED")
            .ok()
            .and_then(|v| v.parse().ok())
            .unwrap_or(DEFAULT_GPU_MIN_BATCH_COMPRESSED)
    })
}
