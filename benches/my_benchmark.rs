use criterion::{black_box, criterion_group, criterion_main, Criterion};
use gda_core::compute::tensor::{Tensor, TensorError};
use gda_core::compute::processor::{Processor, ProcessorSelectionConstraint, Compiled};
const VEC_SIZE: usize = 60000;

fn benchmark_gpu<'a>(shader: &'a mut Compiled<'a, f32>, gpu: &mut Processor) {
    gpu.execute(shader);
}


fn benchmark_cpu(a: &Vec<f32>, b: &Vec<f32>) {
    let mut c = Vec::<f32>::new();

    for i in 0..VEC_SIZE {
        c.push(a[i] * b[i]);
    }
}

fn criterion_benchmark(c: &mut Criterion) {
    let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![VEC_SIZE]);   

    let mut a = Tensor::new(vec![1f32; VEC_SIZE]);
    let mut b = Tensor::new(vec![2f32; VEC_SIZE]);

    let op = &a + &b;

    let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
    
    let mut shader = gpu.build(op, result_tensor);
    

    let a1 = vec![1f32; VEC_SIZE];
    let b1 = vec![2f32; VEC_SIZE];
    {
        c.bench_function("CPU", |b| b.iter(|| benchmark_cpu(black_box(&a1), black_box(&b1))));
        c.bench_function("GPU", |b| b.iter(|| benchmark_gpu(&mut shader, &mut gpu)));
    }
}

criterion_group!(benches, criterion_benchmark);
criterion_main!(benches);