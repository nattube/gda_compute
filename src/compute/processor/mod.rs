pub mod gpu;
use gpu::GPU;
use gpu::shader::Shader;
use crate::compute::tensor::{Operation, SupportedDataTypes, Tensor, TensorError, TensorOperationResult};
use std::cell::RefCell;
use std::rc::Rc;

pub enum Compiled<'a, T> 
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
    GPU(Rc<RefCell<Shader<'a, T>>>)
}

pub enum AbstractProcessor {
    GPU(Box<GPU>)
}

impl AbstractProcessor {
    fn build<'a, T>(&mut self, op: Operation<'a>, result: Tensor<T>) -> Compiled<'a, T>
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        match self {
            AbstractProcessor::GPU(x) => x.build(op, result)
        }
    }
    fn execute<'a, T>(&mut self, compiled: &Compiled<'a, T>) -> TensorOperationResult
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        match self {
            AbstractProcessor::GPU(x) => x.execute(compiled)
        }
    }
}

pub enum ProcessorSelectionConstraint {
    CPU,
    GPU,
    None,
    SelectGPUById(u32)
}

pub struct Processor {
    processor: AbstractProcessor
}

impl Processor {
    pub fn new(choose: ProcessorSelectionConstraint) -> Processor {
        Processor {processor: AbstractProcessor::GPU(Box::new(GPU::new()))}
    }

    pub fn list_available() {
        unimplemented!("Missing");
    }

    pub fn build<'a, T>(&mut self,  op: Operation<'a>, result: Tensor<T>) -> Compiled<'a, T> 
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        self.processor.build(op, result)
    }

    pub fn execute<'a, T>(&mut self, compiled: &Compiled<'a, T>) -> Result<Tensor<T>, TensorError> 
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        let intermediate = self.processor.execute(compiled);
        
        if let TensorOperationResult::Error(x) = intermediate {
            return Err(x)
        }

        let result_shape = match compiled {
            Compiled::GPU(x) => x.borrow().result_tensor.get_shape().to_vec()
        };

        let res: Result<Tensor<T>, TensorError> = T::get_tensor(intermediate, &result_shape);

        res
    }
}