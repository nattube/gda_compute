pub mod gpu;
use gpu::GPU;
use gpu::shader::Shader;
use crate::compute::tensor::{Operation, TensorOperationResult};
use std::cell::RefCell;
use std::rc::Rc;

pub enum Compiled<'a> {
    GPU(Rc<RefCell<Shader<'a>>>)
}

pub trait AbstractProcessor<'a> {
    fn build(&mut self, op: Operation<'a>) -> Compiled<'a>;
    fn execute(&mut self, compiled: &Compiled<'a>) -> TensorOperationResult;
}

pub enum ProcessorSelectionConstraint {
    CPU,
    GPU,
    None,
    SelectGPUById(u32)
}

pub struct Processor<'a> {
    processor: Box<AbstractProcessor<'a>>
}

impl<'a> Processor<'a> {
    pub fn new(choose: ProcessorSelectionConstraint) -> Processor<'a> {
        Processor {processor: Box::new(GPU::new())}
    }

    pub fn list_available() {
        unimplemented!("Missing");
    }

    pub fn build(&mut self,  op: Operation<'a>) -> Compiled<'a> {
        self.processor.build(op)
    }

    pub fn execute(&mut self, compiled: &Compiled<'a>) -> TensorOperationResult {
        self.processor.execute(compiled)
    }
}