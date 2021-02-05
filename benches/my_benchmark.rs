use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gda_core::compute::gpu::*;
use gda_core::compute::gpu::processor::*;

fn benchmarkGPU(shader: &Shader) {
    
    let mut gpu = GPU::new();

    (*shader).execute(&mut gpu);
}


fn benchmarkCPU() {
    let mut a = vec![1u32; 60000];
    let mut b = vec![2u32; 60000];

    let mut c = Vec::<u32>::new();

    for i in 0..60000 {
        c.push(a[i] + b[i]);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let mut a = Tensor::new(vec![1; 60000]);
    let mut b = Tensor::new(vec![2; 60000]);

    let op = &a + &b;
    let shader = Shader::build(&op);

    c.bench_function("CPU", |b| b.iter(|| benchmarkCPU()));
    c.bench_function("GPU", |b| b.iter(|| benchmarkGPU(&shader)));
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);