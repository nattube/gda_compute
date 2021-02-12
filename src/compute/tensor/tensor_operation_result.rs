use std::cell::RefCell;
use std::cell::Ref;
use std::rc::Rc;
use std::ops::{Add};
use std::cmp;
use std::fmt;
use std::fmt::Debug;
use std::convert::TryInto;
use wgpu;
use wgpu::BufferView;

use super::{Operation, SupportedDataTypes, Shape, Tensor, TensorError, TensorOperationResult, TwoValueOperation};


impl TensorOperationResult {
    pub(crate) fn from_2<T, U>(t1: &Tensor<T>, t2: &Tensor<U>, op: TwoValueOperation) -> Self
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T>,
          U: SupportedDataTypes + SupportedDataTypes<BindingType = U> {

        let shape1 = &*t1.get_shape();
        let shape2 = &*t2.get_shape();
        
        let result_shape = TensorOperationResult::get_result_shape(shape1, shape2, &op);

        if let Err(x) = result_shape {
            return TensorOperationResult::Error(x)
        }

        let shape = result_shape.unwrap();
        let m = cmp::max(T::strength(), U::strength());

        match m {
            0 => TensorOperationResult::Int(Box::new(Tensor::with_shape(vec![0i32; shape.iter().sum()], shape))),
            1 => TensorOperationResult::UInt(Box::new(Tensor::with_shape(vec![0u32; shape.iter().sum()], shape))),
            2 => TensorOperationResult::Float(Box::new(Tensor::with_shape(vec![0f32; shape.iter().sum()], shape))),
            3 => TensorOperationResult::Double(Box::new(Tensor::with_shape(vec![0f64; shape.iter().sum()], shape))),
            _ => TensorOperationResult::Error(TensorError::Unimplemented("Don't know type".to_string()))
        }
    }

    pub(crate) fn from_2_ops(operation1: &Operation, operation2: &Operation, op: TwoValueOperation) -> Self {
        let shape1 =  match operation2 {
            Operation::DualOp {result, ..} => {
                result.get_own_shape()
            }
            _ => Shape::new()
        };

        let shape2 = match operation2 {
            Operation::DualOp {result, ..} => {
                result.get_own_shape()
            }
            _ => Shape::new()
        };
        
        let result_shape = TensorOperationResult::get_result_shape(&shape1, &shape2, &op);

        if let Err(x) = result_shape {
            return TensorOperationResult::Error(x)
        }

        let op1_strength = match operation1 {
            Operation::DualOp {result, ..} => {
                result.get_own_strength()
            }
            _ => usize::MAX
        };

        let op2_strength = match operation2 {
            Operation::DualOp {result, ..} => {
                result.get_own_strength()
            }
            _ => usize::MAX
        };

        let shape = result_shape.unwrap();
        let m = cmp::max(op1_strength, op2_strength);

        match m {
            0 => TensorOperationResult::Int(Box::new(Tensor::with_shape(vec![0i32; shape.iter().sum()], shape))),
            1 => TensorOperationResult::UInt(Box::new(Tensor::with_shape(vec![0u32; shape.iter().sum()], shape))),
            2 => TensorOperationResult::Float(Box::new(Tensor::with_shape(vec![0f32; shape.iter().sum()], shape))),
            3 => TensorOperationResult::Double(Box::new(Tensor::with_shape(vec![0f64; shape.iter().sum()], shape))),
            _ => TensorOperationResult::Error(TensorError::Unimplemented("Don't know type".to_string()))
        }
    }

    pub(crate) fn from_1_and_op<T>(t1: &Tensor<T>, operation: &Operation, op: TwoValueOperation) -> Self
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {

        let shape1 = &*t1.get_shape();
        let shape2 = match operation {
            Operation::DualOp {result, ..} => {
                result.get_own_shape()
            }
            _ => Shape::new()
        };
        
        let result_shape = TensorOperationResult::get_result_shape(shape1, &shape2, &op);

        if let Err(x) = result_shape {
            return TensorOperationResult::Error(x)
        }

        let op_strength = match operation {
            Operation::DualOp {result, ..} => {
                result.get_own_strength()
            }
            _ => usize::MAX
        };

        let shape = result_shape.unwrap();
        let m = cmp::max(T::strength(), op_strength);

        match m {
            0 => TensorOperationResult::Int(Box::new(Tensor::with_shape(vec![0i32; shape.iter().sum()], shape))),
            1 => TensorOperationResult::UInt(Box::new(Tensor::with_shape(vec![0u32; shape.iter().sum()], shape))),
            2 => TensorOperationResult::Float(Box::new(Tensor::with_shape(vec![0f32; shape.iter().sum()], shape))),
            3 => TensorOperationResult::Double(Box::new(Tensor::with_shape(vec![0f64; shape.iter().sum()], shape))),
            _ => TensorOperationResult::Error(TensorError::Unimplemented("Don't know type".to_string()))
        }
    }

    fn get_result_shape(s1: &Shape, s2: &Shape, op: &TwoValueOperation) -> Result<Shape, TensorError> {
        match (op) {
            TwoValueOperation::Add => {
                if s1.len() == 1 && s1[0] == 1 {return Ok(s2.to_vec())}
                if s2.len() == 1 && s2[0] == 1  {return Ok(s1.to_vec())}
                if s1.len() != s2.len() {return Err(TensorError::ShapeError("Add shapes not matching".to_string()))}
                if s1.iter().zip(s2).any(|(a, b)| a != b) {return Err(TensorError::ShapeError("Add shapes not matching".to_string()))}
                Ok(s1.to_vec())
            },
            TwoValueOperation::Subtract => {
                if s1.len() == 1 && s1[0] == 1 {return Ok(s2.to_vec())}
                if s2.len() == 1 && s2[0] == 1  {return Ok(s1.to_vec())}
                if s1.len() != s2.len() {return Err(TensorError::ShapeError("Add shapes not matching".to_string()))}
                if s1.iter().zip(s2).any(|(a, b)| a != b) {return Err(TensorError::ShapeError("Add shapes not matching".to_string()))}
                Ok(s1.to_vec())
            },
            _ =>  Err(TensorError::ShapeError("Unknown shape for unknown operation".to_string()))
        }
    }

    fn get_own_strength(&self) -> usize {
        match self {
            TensorOperationResult::Int(..) => 0,
            TensorOperationResult::UInt(..) => 1,
            TensorOperationResult::Float(..) => 2,
            TensorOperationResult::Double(..) => 3,
            _ => usize::MAX
        }
    }

    fn get_own_shape(&self) -> Shape {
        match self {
            TensorOperationResult::Int(x) => (&x).get_shape().to_vec(),
            TensorOperationResult::UInt(x) => (&x).get_shape().to_vec(),
            TensorOperationResult::Float(x) => (&x).get_shape().to_vec(),
            TensorOperationResult::Double(x) => (&x).get_shape().to_vec(),
            _ => Shape::new()
        }
    }

    pub(crate) fn copy(&self) -> Self {
        match self {
            TensorOperationResult::Int(x) => TensorOperationResult::Int(Box::new(x.copy())),
            TensorOperationResult::UInt(x) => TensorOperationResult::UInt(Box::new(x.copy())),
            TensorOperationResult::Float(x) => TensorOperationResult::Float(Box::new(x.copy())),
            TensorOperationResult::Double(x) => TensorOperationResult::Double(Box::new(x.copy())),
            TensorOperationResult::Error(x) => TensorOperationResult::Error(TensorError::Unimplemented("Copy Error Unimplemented".to_string())),
        }
    }

    pub(crate) fn map_from_staging_buffer<'a>(&self, data: BufferView<'a>) {
        match self {
            TensorOperationResult::Int(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorOperationResult::UInt(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorOperationResult::Float(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorOperationResult::Double(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            _ => panic!("Unimplemented Error Handling")
        };
    }

    pub(crate) fn get_mem_size(&self) -> wgpu::BufferAddress {
        (match self {
            TensorOperationResult::Int(x) => x.shape_len() * std::mem::size_of::<i32>(),            
            TensorOperationResult::UInt(x) => x.shape_len() * std::mem::size_of::<u32>(),            
            TensorOperationResult::Float(x) => x.shape_len() * std::mem::size_of::<f32>(),            
            TensorOperationResult::Double(x) => x.shape_len() * std::mem::size_of::<f64>(), 
            _ => panic!("Unimplemented Error Handling")          
        }) as wgpu::BufferAddress
    }

    pub(crate) fn is_single(&self) -> bool {
        match self {
            TensorOperationResult::Int(x) => x.is_single(),
            TensorOperationResult::UInt(x) => x.is_single(),
            TensorOperationResult::Float(x) => x.is_single(),
            TensorOperationResult::Double(x) => x.is_single(),
            _ => panic!("Unimplemented Error Handling")
        }
    }

    pub(crate) fn get_type_glsl(&self) -> String {
        match self {
            TensorOperationResult::Int(x) => if x.is_single() {"int"} else {"int[]"},
            TensorOperationResult::UInt(x) => if x.is_single() {"uint"} else {"uint[]"},
            TensorOperationResult::Float(x) => if x.is_single() {"float"} else {"float[]"},
            TensorOperationResult::Double(x) => if x.is_single() {"double"} else {"double[]"},
            _ => panic!("Unimplemented Error Handling")
        }.to_string()
    }
}

impl Debug for TensorOperationResult {    
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            TensorOperationResult::Int(x) => f.write_fmt(format_args!("{:?}" ,(x))),
            TensorOperationResult::UInt(x) => f.write_fmt(format_args!("{:?}" ,(x))),
            TensorOperationResult::Float(x) => f.write_fmt(format_args!("{:?}" ,(x))),
            TensorOperationResult::Double(x) => f.write_fmt(format_args!("{:?}" ,(x))),
            TensorOperationResult::Error(x) => f.write_str("Error not yet printable")
        }
    }    
}