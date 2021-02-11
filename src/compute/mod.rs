pub mod processor;
pub mod tensor;

use tensor::{Tensor, TensorError};
use processor::{Processor, ProcessorSelectionConstraint};

pub fn run() -> Vec<u32> {
    build_shader();
    //gpu::public_run()
    Vec::new()
}

pub fn build_shader() {
    let a = Tensor::new(vec![1f32,1.0,1.0]);
    let b = Tensor::new(vec![1f32,1.0,1.0]);
    let op = &a + &b;
    let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
    let shader = gpu.build(op);
    let res1 = gpu.execute(&shader);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res1);

    if let Err(TensorError::ShapeError(x)) = b.change_value(vec![4f32,3.0,2.0]) {
        println!("{}", x);
    }
    let res2 = gpu.execute(&shader);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res2);
}