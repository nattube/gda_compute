use std::cell::RefCell;
use std::cell::Ref;
use std::rc::Rc;
use std::ops::{Add};
use std::fmt;
use std::fmt::Debug;

use super::{ Operation, SupportedDataTypes, Shape, Tensor, TensorBinding, TensorError, TensorOperationResult, TwoValueOperation};

impl<T> Tensor<T>
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
    pub fn new(vec: Vec<T>) -> Tensor<T> {
        let shape = vec![vec.len(); 1];
        Tensor {value: RefCell::new(vec), change: Rc::new(RefCell::new(0)), shape: RefCell::new(shape)}
    }

    pub fn with_shape(vec: Vec<T>, shape: Shape) -> Tensor<T> {
        Tensor {value: RefCell::new(vec), change: Rc::new(RefCell::new(0)), shape: RefCell::new(shape)}
    }

    pub fn from_shape_and_value(value: T, shape: Vec<usize>) -> Tensor<T> {
        let vec = vec![value; shape.iter().sum()];
        Tensor {value: RefCell::new(vec), change: Rc::new(RefCell::new(0)), shape: RefCell::new(shape)}
    }

    pub fn zeros_from_shape(shape: Vec<usize>) -> Tensor<T> {
        Self::from_shape_and_value(T::get_zero(), shape)
    }

    pub fn is_single(&self) -> bool {
        self.shape_len() == 1
    }

    pub fn shape_len(&self) -> usize {
        self.shape.borrow().iter().sum()
    }

    pub fn get_value(&self) -> Ref<Vec<T>> {
        self.value.borrow()
    }

    pub fn get_shape(&self) -> Ref<Shape> {
        self.shape.borrow()
    }

    pub fn change_value(&self, mut val: Vec<T>) -> Result<(), TensorError> {
        let target_length = self.shape_len();
        let actual_length = val.len();
        
        if target_length != actual_length {
            return Err(TensorError::ShapeError(
                format!("Tensor Shape requires a vector with {} elements, but new Vector had {} elements! Consider Changing the Shape", target_length, actual_length)
            ))
        }

        let mut old_val = self.value.borrow_mut();
        old_val.clear();
        old_val.append(&mut val);
        *self.change.borrow_mut() += 1;

        Ok(())
    }

    pub(crate) fn get_change(&self) -> u32 {
        *self.change.borrow()
    }

    pub(crate) fn copy(&self) -> Tensor<T> {
        Tensor::with_shape(self.get_value().to_vec(), self.get_shape().to_vec())
    }

    pub(crate) fn same_shape_as<U>(&self, other: &Tensor<U>) -> bool 
    where U: SupportedDataTypes + SupportedDataTypes<BindingType = U> {
        let s1 = &*self.get_shape();
        let s2 = &*other.get_shape();

        if s1.len() != s2.len() {return false}
        if s1.iter().zip(s2).any(|(x,y)| x != y) {return false}

        true
    }

    pub(crate) fn matches_shape(&self, s2: &Shape) -> bool  {
        let s1 = &*self.get_shape();

        if s1.len() != s2.len() {return false}
        if s1.iter().zip(s2).any(|(x,y)| x != y) {return false}

        true
    }
}

impl<'a, T, U> Add<&'a Tensor<U>> for &'a Tensor<T>
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T>,
      U: SupportedDataTypes + SupportedDataTypes<BindingType = U> {
    type Output = Operation<'a>;

    fn add(self, other: &'a Tensor<U>) -> Operation<'a> {
        Operation::DualOp {
            left: Box::new(Operation::Var(Box::new(TensorBinding::from_tensor(self, 0u32)))), 
            right: Box::new(Operation::Var(Box::new(TensorBinding::from_tensor(other, 1u32)))),
            result: TensorOperationResult::from_2(self, other, TwoValueOperation::Add),
            op: TwoValueOperation::Add
        }
    }
}

impl<'a, T> Add<Operation<'a>> for &'a Tensor<T>
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
    type Output = Operation<'a>;

    fn add(self, op: Operation<'a>) -> Operation<'a> {
        let result = TensorOperationResult::from_1_and_op(self, &op, TwoValueOperation::Add);
        let binding = op.get_last_binding() + 1;

        let tensor = match op.contains_tensor(self) {
            Some(x) => x.copy(),
            None => TensorBinding::from_tensor(self, binding)
        };

        Operation::DualOp {
            left: Box::new(Operation::Var(Box::new(tensor))), 
            right: Box::new(op),
            result,
            op: TwoValueOperation::Add
        }
    }
}

impl<T> Debug for Tensor<T>
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> + std::fmt::Display, 
      Vec<T>: Debug {    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        if self.is_single() {
            f.write_fmt(format_args!("{}" ,(*self.get_value())[0]))
        }
        else {
            f.write_fmt(format_args!("{:?}", self.get_value()))
        }
    }    
}