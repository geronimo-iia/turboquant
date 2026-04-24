use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Lazily-initialized GPU context. Created once, reused across calls.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    score_pipeline: wgpu::ComputePipeline,
    bind_group_layout: wgpu::BindGroupLayout,
}

static GPU_CONTEXT: OnceLock<Option<GpuContext>> = OnceLock::new();

impl GpuContext {
    /// Try to initialize GPU. Returns None if no adapter found.
    pub fn try_init() -> Option<Self> {
        let instance = wgpu::Instance::new(wgpu::InstanceDescriptor {
            backends: wgpu::Backends::all(),
            ..wgpu::InstanceDescriptor::new_without_display_handle()
        });

        let adapter =
            match pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                compatible_surface: None,
                force_fallback_adapter: false,
            })) {
                Ok(a) => a,
                Err(e) => {
                    log::warn!("GPU adapter not available ({e}), falling back to CPU");
                    return None;
                }
            };

        let (device, queue) =
            match pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: Some("qjl-sketch"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                ..Default::default()
            })) {
                Ok(dq) => dq,
                Err(e) => {
                    log::warn!("GPU device request failed ({e}), falling back to CPU");
                    return None;
                }
            };

        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("score_compressed"),
            source: wgpu::ShaderSource::Wgsl(include_str!("score.wgsl").into()),
        });

        // 7 bindings: 1 uniform + 5 storage(read) + 1 storage(rw)
        let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("score_compressed_layout"),
            entries: &[
                bgl_entry(0, wgpu::BufferBindingType::Uniform),
                bgl_entry(1, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(2, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(3, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(4, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(5, wgpu::BufferBindingType::Storage { read_only: true }),
                bgl_entry(6, wgpu::BufferBindingType::Storage { read_only: false }),
            ],
        });

        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("score_compressed_pipeline_layout"),
            bind_group_layouts: &[Some(&bind_group_layout)],
            immediate_size: 0,
        });

        let score_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("score_compressed_pipeline"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });

        Some(Self {
            device,
            queue,
            score_pipeline,
            bind_group_layout,
        })
    }

    /// Global singleton (lazy, thread-safe).
    pub fn get() -> Option<&'static Self> {
        GPU_CONTEXT.get_or_init(GpuContext::try_init).as_ref()
    }

    /// Score compressed vectors on GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn score_compressed_batch(
        &self,
        a_inlier_bytes: &[u8],
        b_inlier_bytes: &[u8],
        a_outlier_bytes: &[u8],
        b_outlier_bytes: &[u8],
        a_norms: &[f32],
        b_norms: &[f32],
        a_outlier_norms: &[f32],
        b_outlier_norms: &[f32],
        num_pairs: usize,
        sketch_dim: usize,
        outlier_sketch_dim: usize,
    ) -> Vec<f32> {
        if num_pairs == 0 {
            return Vec::new();
        }

        let inlier_words_per_vec = sketch_dim / 32;
        let outlier_words_per_vec = outlier_sketch_dim / 32;

        let a_inlier_u32 = pack_bytes_to_u32(a_inlier_bytes);
        let b_inlier_u32 = pack_bytes_to_u32(b_inlier_bytes);
        let a_outlier_u32 = pack_bytes_to_u32(a_outlier_bytes);
        let b_outlier_u32 = pack_bytes_to_u32(b_outlier_bytes);

        // Pack norms: [a_norm, a_out_norm, b_norm, b_out_norm] × num_pairs
        let mut packed_norms = Vec::with_capacity(num_pairs * 4);
        for i in 0..num_pairs {
            packed_norms.push(a_norms[i]);
            packed_norms.push(a_outlier_norms[i]);
            packed_norms.push(b_norms[i]);
            packed_norms.push(b_outlier_norms[i]);
        }

        let params = [
            sketch_dim as u32,
            outlier_sketch_dim as u32,
            inlier_words_per_vec as u32,
            outlier_words_per_vec as u32,
            num_pairs as u32,
            0,
            0,
            0,
        ];

        let params_buf = self.create_buffer_init("params", bytemuck::cast_slice(&params), true);
        let a_inlier_buf =
            self.create_buffer_init("a_inlier", bytemuck::cast_slice(&a_inlier_u32), false);
        let b_inlier_buf =
            self.create_buffer_init("b_inlier", bytemuck::cast_slice(&b_inlier_u32), false);
        let a_outlier_buf =
            self.create_buffer_init("a_outlier", bytemuck::cast_slice(&a_outlier_u32), false);
        let b_outlier_buf =
            self.create_buffer_init("b_outlier", bytemuck::cast_slice(&b_outlier_u32), false);
        let norms_buf =
            self.create_buffer_init("norms", bytemuck::cast_slice(&packed_norms), false);

        let scores_size = (num_pairs * 4) as u64;
        let scores_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("scores"),
            size: scores_size,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });

        let readback_buf = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("readback"),
            size: scores_size,
            usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("score_bind_group"),
            layout: &self.bind_group_layout,
            entries: &[
                bg_entry(0, &params_buf),
                bg_entry(1, &a_inlier_buf),
                bg_entry(2, &b_inlier_buf),
                bg_entry(3, &a_outlier_buf),
                bg_entry(4, &b_outlier_buf),
                bg_entry(5, &norms_buf),
                bg_entry(6, &scores_buf),
            ],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("score_encoder"),
            });

        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("score_pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.score_pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_pairs as u32).div_ceil(64), 1, 1);
        }

        encoder.copy_buffer_to_buffer(&scores_buf, 0, &readback_buf, 0, scores_size);
        self.queue.submit(Some(encoder.finish()));

        let slice = readback_buf.slice(..);
        let (tx, rx) = std::sync::mpsc::channel();
        slice.map_async(wgpu::MapMode::Read, move |result| {
            tx.send(result).unwrap();
        });
        self.device
            .poll(wgpu::PollType::Wait {
                submission_index: None,
                timeout: Some(std::time::Duration::from_secs(10)),
            })
            .unwrap();
        rx.recv().unwrap().unwrap();

        let data = slice.get_mapped_range();
        let scores: Vec<f32> = bytemuck::cast_slice(&data).to_vec();
        drop(data);
        readback_buf.unmap();

        scores
    }

    fn create_buffer_init(&self, label: &str, data: &[u8], uniform: bool) -> wgpu::Buffer {
        let usage = if uniform {
            wgpu::BufferUsages::UNIFORM
        } else {
            wgpu::BufferUsages::STORAGE
        };
        self.device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some(label),
                contents: data,
                usage,
            })
    }
}

fn pack_bytes_to_u32(bytes: &[u8]) -> Vec<u32> {
    let padded_len = bytes.len().div_ceil(4) * 4;
    let mut padded = vec![0u8; padded_len];
    padded[..bytes.len()].copy_from_slice(bytes);
    padded
        .chunks_exact(4)
        .map(|c| u32::from_le_bytes([c[0], c[1], c[2], c[3]]))
        .collect()
}

fn bgl_entry(binding: u32, ty: wgpu::BufferBindingType) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn bg_entry(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[ignore] // requires GPU adapter
    fn gpu_context_initializes() {
        let ctx = GpuContext::try_init();
        assert!(ctx.is_some(), "no GPU adapter found");
    }

    #[test]
    #[ignore] // requires GPU adapter
    fn gpu_score_matches_cpu() {
        use crate::score::hamming_similarity;

        let ctx = GpuContext::try_init().expect("no GPU adapter");
        let sketch_dim = 256usize;
        let outlier_sketch_dim = 64usize;
        let num_pairs = 100;
        let inlier_bytes = sketch_dim / 8;
        let outlier_bytes = outlier_sketch_dim / 8;

        let mut a_inlier = vec![0u8; num_pairs * inlier_bytes];
        let mut b_inlier = vec![0u8; num_pairs * inlier_bytes];
        let mut a_outlier = vec![0u8; num_pairs * outlier_bytes];
        let mut b_outlier = vec![0u8; num_pairs * outlier_bytes];
        let mut a_norms = vec![0.0f32; num_pairs];
        let mut b_norms = vec![0.0f32; num_pairs];
        let mut a_out_norms = vec![0.0f32; num_pairs];
        let mut b_out_norms = vec![0.0f32; num_pairs];

        for i in 0..num_pairs {
            a_norms[i] = 1.0 + (i as f32) * 0.01;
            b_norms[i] = 1.0 + (i as f32) * 0.02;
            a_out_norms[i] = 0.1 + (i as f32) * 0.001;
            b_out_norms[i] = 0.1 + (i as f32) * 0.002;
            for j in 0..inlier_bytes {
                a_inlier[i * inlier_bytes + j] = ((i + j) % 256) as u8;
                b_inlier[i * inlier_bytes + j] = ((i + j + 1) % 256) as u8;
            }
            for j in 0..outlier_bytes {
                a_outlier[i * outlier_bytes + j] = ((i * 3 + j) % 256) as u8;
                b_outlier[i * outlier_bytes + j] = ((i * 3 + j + 2) % 256) as u8;
            }
        }

        let gpu_scores = ctx.score_compressed_batch(
            &a_inlier,
            &b_inlier,
            &a_outlier,
            &b_outlier,
            &a_norms,
            &b_norms,
            &a_out_norms,
            &b_out_norms,
            num_pairs,
            sketch_dim,
            outlier_sketch_dim,
        );

        for i in 0..num_pairs {
            let ai = &a_inlier[i * inlier_bytes..(i + 1) * inlier_bytes];
            let bi = &b_inlier[i * inlier_bytes..(i + 1) * inlier_bytes];
            let ao = &a_outlier[i * outlier_bytes..(i + 1) * outlier_bytes];
            let bo = &b_outlier[i * outlier_bytes..(i + 1) * outlier_bytes];

            let sim_inlier = hamming_similarity(ai, bi, sketch_dim);
            let cos_inlier = (std::f32::consts::PI * (1.0 - sim_inlier)).cos();
            let in_a = (a_norms[i] * a_norms[i] - a_out_norms[i] * a_out_norms[i])
                .max(0.0)
                .sqrt();
            let in_b = (b_norms[i] * b_norms[i] - b_out_norms[i] * b_out_norms[i])
                .max(0.0)
                .sqrt();

            let sim_outlier = hamming_similarity(ao, bo, outlier_sketch_dim);
            let cos_outlier = (std::f32::consts::PI * (1.0 - sim_outlier)).cos();

            let cpu_score =
                in_a * in_b * cos_inlier + a_out_norms[i] * b_out_norms[i] * cos_outlier;

            assert!(
                (gpu_scores[i] - cpu_score).abs() < 1e-3,
                "pair {i}: gpu={}, cpu={cpu_score}, diff={}",
                gpu_scores[i],
                (gpu_scores[i] - cpu_score).abs()
            );
        }
    }

    #[test]
    #[ignore] // requires GPU adapter
    fn gpu_score_empty_input() {
        let ctx = GpuContext::try_init().expect("no GPU adapter");
        let scores = ctx.score_compressed_batch(&[], &[], &[], &[], &[], &[], &[], &[], 0, 256, 64);
        assert!(scores.is_empty());
    }
}
