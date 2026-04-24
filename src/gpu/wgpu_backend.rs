use std::sync::OnceLock;
use wgpu::util::DeviceExt;

/// Lazily-initialized GPU context with float×sign compute pipeline.
pub struct GpuContext {
    device: wgpu::Device,
    queue: wgpu::Queue,
    float_sign_pipeline: wgpu::ComputePipeline,
    float_sign_layout: wgpu::BindGroupLayout,
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
            label: Some("score_float_sign"),
            source: wgpu::ShaderSource::Wgsl(include_str!("score_float_sign.wgsl").into()),
        });
        let float_sign_layout = create_layout(&device, "float_sign", 7);
        let float_sign_pipeline =
            create_pipeline(&device, "float_sign", &float_sign_layout, &shader);

        Some(Self {
            device,
            queue,
            float_sign_pipeline,
            float_sign_layout,
        })
    }

    /// Global singleton (lazy, thread-safe).
    pub fn get() -> Option<&'static Self> {
        GPU_CONTEXT.get_or_init(GpuContext::try_init).as_ref()
    }

    /// Score a query against compressed keys on GPU (float×sign).
    #[allow(clippy::too_many_arguments)]
    pub fn score_float_sign_batch(
        &self,
        q_inlier_sketch: &[f32],
        q_outlier_sketch: &[f32],
        key_quant: &[u8],
        key_outlier_quant: &[u8],
        key_norms: &[f32],
        outlier_norms: &[f32],
        num_vectors: usize,
        sketch_dim: usize,
        outlier_sketch_dim: usize,
        scale: f32,
        scale_outlier: f32,
    ) -> Vec<f32> {
        if num_vectors == 0 {
            return Vec::new();
        }

        let inlier_words = sketch_dim / 32;
        let outlier_words = outlier_sketch_dim / 32;

        // Pack norms: [key_norm, outlier_norm] x num_vectors
        let mut packed_norms = Vec::with_capacity(num_vectors * 2);
        for i in 0..num_vectors {
            packed_norms.push(key_norms[i]);
            packed_norms.push(outlier_norms[i]);
        }

        // Params: 5 u32 + 2 f32, padded to 32 bytes
        let params_u32: [u32; 8] = [
            sketch_dim as u32,
            outlier_sketch_dim as u32,
            inlier_words as u32,
            outlier_words as u32,
            num_vectors as u32,
            scale.to_bits(),
            scale_outlier.to_bits(),
            0,
        ];

        let buffers = [
            self.create_buf("params", bytemuck::cast_slice(&params_u32), true),
            self.create_buf("q_inlier", bytemuck::cast_slice(q_inlier_sketch), false),
            self.create_buf("q_outlier", bytemuck::cast_slice(q_outlier_sketch), false),
            self.create_buf(
                "key_quant",
                bytemuck::cast_slice(&pack_bytes_to_u32(key_quant)),
                false,
            ),
            self.create_buf(
                "key_outlier",
                bytemuck::cast_slice(&pack_bytes_to_u32(key_outlier_quant)),
                false,
            ),
            self.create_buf("norms", bytemuck::cast_slice(&packed_norms), false),
        ];

        self.dispatch(
            &self.float_sign_pipeline,
            &self.float_sign_layout,
            &buffers,
            num_vectors,
        )
    }

    fn dispatch(
        &self,
        pipeline: &wgpu::ComputePipeline,
        layout: &wgpu::BindGroupLayout,
        input_buffers: &[wgpu::Buffer],
        num_items: usize,
    ) -> Vec<f32> {
        let scores_size = (num_items * 4) as u64;
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

        let mut entries: Vec<wgpu::BindGroupEntry<'_>> = input_buffers
            .iter()
            .enumerate()
            .map(|(i, buf)| bg_entry(i as u32, buf))
            .collect();
        entries.push(bg_entry(input_buffers.len() as u32, &scores_buf));

        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("bind_group"),
            layout,
            entries: &entries,
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("encoder"),
            });
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups((num_items as u32).div_ceil(64), 1, 1);
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

    fn create_buf(&self, label: &str, data: &[u8], uniform: bool) -> wgpu::Buffer {
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

fn create_layout(device: &wgpu::Device, label: &str, num_bindings: u32) -> wgpu::BindGroupLayout {
    let mut entries = Vec::with_capacity(num_bindings as usize);
    entries.push(bgl_entry(0, wgpu::BufferBindingType::Uniform));
    for i in 1..num_bindings - 1 {
        entries.push(bgl_entry(
            i,
            wgpu::BufferBindingType::Storage { read_only: true },
        ));
    }
    entries.push(bgl_entry(
        num_bindings - 1,
        wgpu::BufferBindingType::Storage { read_only: false },
    ));
    device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some(label),
        entries: &entries,
    })
}

fn create_pipeline(
    device: &wgpu::Device,
    label: &str,
    layout: &wgpu::BindGroupLayout,
    shader: &wgpu::ShaderModule,
) -> wgpu::ComputePipeline {
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some(label),
        bind_group_layouts: &[Some(layout)],
        immediate_size: 0,
    });
    device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some(label),
        layout: Some(&pipeline_layout),
        module: shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    })
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
    fn gpu_float_sign_score_matches_cpu() {
        use crate::sketch::QJLSketch;
        use rand::SeedableRng;
        use rand_chacha::ChaCha20Rng;
        use rand_distr::{Distribution, StandardNormal};

        let ctx = GpuContext::try_init().expect("no GPU adapter");
        let d = 64;
        let s = 256;
        let os = 64;
        let sketch = QJLSketch::new(d, s, os, 42).unwrap();
        let mut rng = ChaCha20Rng::seed_from_u64(123);
        let normal: StandardNormal = StandardNormal;

        let query: Vec<f32> = (0..d)
            .map(|_| {
                let v: f64 = normal.sample(&mut rng);
                v as f32
            })
            .collect();
        let num_vectors = 100;
        let keys: Vec<f32> = (0..num_vectors * d)
            .map(|_| {
                let v: f64 = normal.sample(&mut rng);
                v as f32
            })
            .collect();

        let outlier_indices = vec![0u8];
        let compressed = sketch
            .quantize(&keys, num_vectors, &outlier_indices)
            .unwrap();

        // CPU scores
        let cpu_scores = sketch.score(&query, &compressed).unwrap();

        // GPU: compute query sketches on CPU (same as score() does)
        let q_sketch = crate::sketch::matvec(&sketch.proj_dir_quant, s, d, &query);
        let mut q_outlier_sketch = vec![0.0f32; s];
        for &idx in &compressed.outlier_indices {
            let j = idx as usize;
            let row_start = j * s;
            for (p, qos) in q_outlier_sketch.iter_mut().enumerate().take(s) {
                *qos += query[j] * sketch.proj_dir_score[row_start + p];
            }
        }
        let q_inlier_sketch: Vec<f32> = q_sketch
            .iter()
            .zip(q_outlier_sketch.iter())
            .map(|(f, o)| f - o)
            .collect();

        let scale = (std::f32::consts::FRAC_PI_2).sqrt() / s as f32;
        let scale_outlier = (std::f32::consts::FRAC_PI_2).sqrt() / os as f32;

        let gpu_scores = ctx.score_float_sign_batch(
            &q_inlier_sketch,
            &q_outlier_sketch[..os],
            &compressed.key_quant,
            &compressed.key_outlier_quant,
            &compressed.key_norms,
            &compressed.outlier_norms,
            num_vectors,
            s,
            os,
            scale,
            scale_outlier,
        );

        for i in 0..num_vectors {
            assert!(
                (gpu_scores[i] - cpu_scores[i]).abs() < 1e-2,
                "vec {i}: gpu={}, cpu={}",
                gpu_scores[i],
                cpu_scores[i]
            );
        }
    }

    #[test]
    #[ignore] // requires GPU adapter
    fn gpu_score_empty_input() {
        let ctx = GpuContext::try_init().expect("no GPU adapter");
        let scores = ctx.score_float_sign_batch(&[], &[], &[], &[], &[], &[], 0, 256, 64, 0.0, 0.0);
        assert!(scores.is_empty());
    }
}
