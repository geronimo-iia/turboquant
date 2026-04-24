// Compressed-vs-compressed scoring via Hamming cosine estimation.
//
// Each thread processes one vector pair.
// Norms are packed into a single buffer: [a_norm, a_out_norm, b_norm, b_out_norm] × num_pairs

struct Params {
    sketch_dim: u32,
    outlier_sketch_dim: u32,
    inlier_words: u32,
    outlier_words: u32,
    num_pairs: u32,
}

@group(0) @binding(0) var<uniform> params: Params;
@group(0) @binding(1) var<storage, read> a_inlier: array<u32>;
@group(0) @binding(2) var<storage, read> b_inlier: array<u32>;
@group(0) @binding(3) var<storage, read> a_outlier: array<u32>;
@group(0) @binding(4) var<storage, read> b_outlier: array<u32>;
// norms: [a_norm_0, a_out_0, b_norm_0, b_out_0, a_norm_1, a_out_1, b_norm_1, b_out_1, ...]
@group(0) @binding(5) var<storage, read> norms: array<f32>;
@group(0) @binding(6) var<storage, read_write> scores: array<f32>;

const PI: f32 = 3.14159265358979323846;

@compute @workgroup_size(64)
fn main(@builtin(global_invocation_id) gid: vec3<u32>) {
    let idx = gid.x;
    if idx >= params.num_pairs {
        return;
    }

    // Inlier: count matching bits
    let inlier_base = idx * params.inlier_words;
    var inlier_matching: u32 = 0u;
    for (var w: u32 = 0u; w < params.inlier_words; w = w + 1u) {
        inlier_matching += countOneBits(~(a_inlier[inlier_base + w] ^ b_inlier[inlier_base + w]));
    }
    let sim_inlier = f32(inlier_matching) / f32(params.sketch_dim);
    let cos_inlier = cos(PI * (1.0 - sim_inlier));

    // Unpack norms (4 floats per pair)
    let norm_base = idx * 4u;
    let a_norm = norms[norm_base];
    let a_out_norm = norms[norm_base + 1u];
    let b_norm = norms[norm_base + 2u];
    let b_out_norm = norms[norm_base + 3u];

    let inlier_norm_a = sqrt(max(a_norm * a_norm - a_out_norm * a_out_norm, 0.0));
    let inlier_norm_b = sqrt(max(b_norm * b_norm - b_out_norm * b_out_norm, 0.0));
    let score_inlier = inlier_norm_a * inlier_norm_b * cos_inlier;

    // Outlier: count matching bits
    var score_outlier: f32 = 0.0;
    if params.outlier_sketch_dim > 0u {
        let outlier_base = idx * params.outlier_words;
        var outlier_matching: u32 = 0u;
        for (var w: u32 = 0u; w < params.outlier_words; w = w + 1u) {
            outlier_matching += countOneBits(~(a_outlier[outlier_base + w] ^ b_outlier[outlier_base + w]));
        }
        let sim_outlier = f32(outlier_matching) / f32(params.outlier_sketch_dim);
        let cos_outlier = cos(PI * (1.0 - sim_outlier));
        score_outlier = a_out_norm * b_out_norm * cos_outlier;
    }

    scores[idx] = score_inlier + score_outlier;
}
