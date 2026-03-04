// Benchmark: Optimized vs Standard Paged Attention
// Run: cargo bench --features metal -- paged_attention_comparison

use criterion::{black_box, criterion_group, criterion_main, Criterion, BenchmarkId};

#[cfg(feature = "metal")]
fn bench_comparison(c: &mut Criterion) {
    use candle_core::{Device, Tensor, DType};

    let device = Device::new_metal(0).expect("Metal device not available");

    let configs: Vec<(&str, usize, usize, usize, usize)> = vec![
        // (name, batch, heads, head_dim, seq_len)
        ("decode_512", 1, 32, 128, 512),
        ("decode_1024", 1, 32, 128, 1024),
        ("decode_2048", 1, 32, 128, 2048),
        ("batch4_512", 4, 32, 128, 512),
        ("batch8_512", 8, 32, 128, 512),
    ];

    let mut group = c.benchmark_group("paged_attention_comparison");

    for (name, batch, heads, head_dim, seq_len) in &configs {
        let q_shape = (*batch, *heads, *seq_len, *head_dim);
        let kv_heads = *heads / 4;
        let kv_shape = (*batch, kv_heads, *seq_len, *head_dim);

        let q = Tensor::randn(0f32, 1f32, q_shape, &device).unwrap()
            .to_dtype(DType::BF16).unwrap();
        let k = Tensor::randn(0f32, 1f32, kv_shape, &device).unwrap()
            .to_dtype(DType::BF16).unwrap();
        let v = Tensor::randn(0f32, 1f32, kv_shape, &device).unwrap()
            .to_dtype(DType::BF16).unwrap();

        let hd = *head_dim;

        // Standard attention
        group.bench_function(BenchmarkId::new("standard", name), |b| {
            b.iter(|| {
                let scale = 1.0f64 / (hd as f64).sqrt();
                let q_scaled = (&q * scale).unwrap();
                let k_t = k.transpose(2, 3).unwrap();
                let scores = q_scaled.matmul(&k_t).unwrap();
                let max_scores = scores.max_keepdim(3).unwrap();
                let exp_scores = (&scores - &max_scores).unwrap().exp().unwrap();
                let sum_exp = exp_scores.sum_keepdim(3).unwrap();
                let attn_weights = (&exp_scores / &sum_exp).unwrap();
                let output = attn_weights.matmul(&v).unwrap();
                black_box(output)
            });
        });

        // Optimized attention (placeholder - same compute for baseline)
        // TODO: Call optimized kernel when implemented
        group.bench_function(BenchmarkId::new("optimized", name), |b| {
            b.iter(|| {
                let scale = 1.0f64 / (hd as f64).sqrt();
                let q_scaled = (&q * scale).unwrap();
                let k_t = k.transpose(2, 3).unwrap();
                let scores = q_scaled.matmul(&k_t).unwrap();
                let max_scores = scores.max_keepdim(3).unwrap();
                let exp_scores = (&scores - &max_scores).unwrap().exp().unwrap();
                let sum_exp = exp_scores.sum_keepdim(3).unwrap();
                let attn_weights = (&exp_scores / &sum_exp).unwrap();
                let output = attn_weights.matmul(&v).unwrap();
                black_box(output)
            });
        });
    }

    group.finish();
}

#[cfg(feature = "metal")]
criterion_group!(benches, bench_comparison);
#[cfg(feature = "metal")]
criterion_main!(benches);

#[cfg(not(feature = "metal"))]
fn main() {
    println!("Metal feature not enabled. Run with --features metal");
}
