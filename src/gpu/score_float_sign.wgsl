// Float×sign scoring: float query sketch × packed sign bits.
//
// Each thread processes one key vector:
//   dot = Σ q_sketch[i] × sign(key_quant[v, i])
// where sign is +1 if bit set, -1 if not.
//
// Norms packed as [key_norm, outlier_norm] × num_vectors in binding 5.

struct Params {
    sketch_dim: u32,
    outlier_sketch_dim: u32,
    inlier_words: u32,
    outlier_words: u32,
    num_vectors: u32,
    scale: f32,          // sqrt(π/2) / sketch_dim
    scale_outlier: f32,  // sqrt(π/2) / outlier_sketch_dim
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> q_inlier_sketch: array<f32>;
@group(0) @binding(2) var<storage, read> q_outlier_sketch: array<f32>;
@group(0) @binding(3) var<storage, read> key_quant: array<u32>;
@group(0) @binding(4) var<storage, read> key_outlier_quant: array<u32>;
// norms: [key_norm, outlier_norm] × num_vectors
@group(0) @binding(5) var<storage, read> norms: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let v = gid.x;
    if v >= params.num_vectors {
        return;
    }

    // Inlier: dot(q_inlier_sketch, sign(key_quant[v]))
    var dot_inlier: f32 = 0.0;
    let inlier_base = v * params.inlier_words;
    for (var w: u32 = 0u; w < params.inlier_words; w = w + 1u) {
        let word = key_quant[inlier_base + w];
        let bit_offset = w * 32u;
        for (var b: u32 = 0u; b < 32u; b = b + 1u) {
            let idx = bit_offset + b;
            if idx >= params.sketch_dim {
                break;
            }
            let sign_bit = (word >> b) & 1u;
            let sign = select(-1.0, 1.0, sign_bit == 1u);
            dot_inlier += q_inlier_sketch[idx] * sign;
        }
    }

    // Outlier: dot(q_outlier_sketch, sign(key_outlier_quant[v]))
    var dot_outlier: f32 = 0.0;
    if params.outlier_sketch_dim > 0u {
        let outlier_base = v * params.outlier_words;
        for (var w: u32 = 0u; w < params.outlier_words; w = w + 1u) {
            let word = key_outlier_quant[outlier_base + w];
            let bit_offset = w * 32u;
            for (var b: u32 = 0u; b < 32u; b = b + 1u) {
                let idx = bit_offset + b;
                if idx >= params.outlier_sketch_dim {
                    break;
                }
                let sign_bit = (word >> b) & 1u;
                let sign = select(-1.0, 1.0, sign_bit == 1u);
                dot_outlier += q_outlier_sketch[idx] * sign;
            }
        }
    }

    // Norms: [key_norm, outlier_norm] per vector
    let norm_base = v * 2u;
    let key_norm = norms[norm_base];
    let outlier_norm = norms[norm_base + 1u];
    let inlier_norm = sqrt(max(key_norm * key_norm - outlier_norm * outlier_norm, 0.0));

    scores[v] = params.scale * inlier_norm * dot_inlier
              + params.scale_outlier * outlier_norm * dot_outlier;
}
