use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gda_core::compute::gpu::*;
use gda_core::compute::gpu::processor::*;

const VEC_SIZE: usize = 60000;

fn benchmark_gpu(shader: &mut Shader, gpu: &mut GPU) {
    (*shader).execute(gpu);
}


fn benchmark_cpu(a: &Vec<f32>, b: &Vec<f32>) {
    let mut c = Vec::<f32>::new();

    for i in 0..VEC_SIZE {
        c.push(a[i] * b[i]);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut a = Tensor::new(vec![1f32; VEC_SIZE]);
    let mut b = Tensor::new(vec![2f32; VEC_SIZE]);

    let op = &a + &b;
    let mut gpu = GPU::new();
    let mut shader = Shader::build(&op, &mut gpu);

    let a1 = vec![1f32; VEC_SIZE];
    let b1 = vec![2f32; VEC_SIZE];

    c.bench_function("CPU", |b| b.iter(|| benchmark_cpu(black_box(&a1), black_box(&b1))));
    c.bench_function("GPU", |b| b.iter(|| benchmark_gpu(&mut shader, &mut gpu)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);