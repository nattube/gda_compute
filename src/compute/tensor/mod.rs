use std::cell::RefCell;
use std::rc::Rc;
use std::fmt;

pub mod operation;
pub mod supported_datatypes;
pub mod tensor;
pub mod tensor_binding;
pub mod tensor_holder;
pub mod tensor_operation_result;

const INPUT_NAME: &str = "inv";

#[derive(Debug)]
pub enum TensorError {
    ShapeError(String),
    Unimplemented(String)
}

pub enum TensorOperationResult {
    Int(Box<Tensor<i32>>),
    UInt(Box<Tensor<u32>>),
    Float(Box<Tensor<f32>>),
    Double(Box<Tensor<f64>>),
    Error(TensorError)
}

pub enum TensorHolder<'a> {
    Int(&'a Tensor<i32>),
    UInt(&'a Tensor<u32>),
    Float(&'a Tensor<f32>),
    Double(&'a Tensor<f64>)
}

pub trait SupportedDataTypes: bytemuck::Pod + Clone {
    type BindingType;
    fn to_data_holder(vec: &Tensor<Self::BindingType>) -> TensorHolder;
    fn strength() -> usize;
    fn get_zero() -> Self::BindingType;
    fn get_tensor(res: TensorOperationResult, wanted_shape: &Shape) -> Result<Tensor<Self::BindingType>, TensorError>;
}

pub struct TensorBinding<'a> {
    pub(crate) id: u32,
    pub(crate) value: TensorHolder<'a>,
    pub(crate) change: Rc<RefCell<u32>>
}

pub type Shape = Vec<usize>;

pub struct Tensor<T> {
    pub(crate) value: RefCell<Vec<T>>,
    pub(crate) change: Rc<RefCell<u32>>,
    pub(crate) shape: RefCell<Shape>,
    pub(crate) is_const: bool
}

pub enum KnownOperation {
    Single(SingleValueOperation),
    Dual(TwoValueOperation)
}

pub enum SingleValueOperation {
    SquareRoot
}

pub enum TwoValueOperation {
    Add,
    Subtract,
    Multiply,
    Divide
}

impl fmt::Debug for TwoValueOperation {    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TwoValueOperation::Add => f.write_str("+"),
            TwoValueOperation::Subtract => f.write_str("-"),
            TwoValueOperation::Multiply => f.write_str("*"),
            TwoValueOperation::Divide => f.write_str("/")
        }
    }    
}

pub enum Operation<'a> {    
    Var(Box<TensorBinding<'a>>),
    SingleOp {value: Box<Operation<'a>>, result: TensorOperationResult, op: SingleValueOperation},
    DualOp {left: Box<Operation<'a>>, right: Box<Operation<'a>>, result: TensorOperationResult, op: TwoValueOperation},
}