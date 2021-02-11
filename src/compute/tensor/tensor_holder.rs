use std::ptr;
use wgpu;
use std::convert::TryInto;
use wgpu::BufferView;

use super::{TensorHolder};

impl<'a> TensorHolder<'a> {
    pub(crate) fn get_change(&self) -> u32 {
        match self {
            TensorHolder::Int(x) => x.get_change(),
            TensorHolder::UInt(x) => x.get_change(),
            TensorHolder::Float(x) => x.get_change(),
            TensorHolder::Double(x) => x.get_change()
        }
    }

    pub(crate) fn map_from_staging_buffer(&self, data: BufferView<'a>) {
        match self {
            TensorHolder::Int(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| i32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorHolder::UInt(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorHolder::Float(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect()),

            TensorHolder::Double(x) => 
                x.change_value(data
                    .chunks_exact(4)
                    .map(|b| f64::from_ne_bytes(b.try_into().unwrap()))
                    .collect())
        };
    }

    pub(crate) fn get_mem_size(&self) -> wgpu::BufferAddress {
        (match self {
            TensorHolder::Int(x) => x.shape_len() * std::mem::size_of::<i32>(),            
            TensorHolder::UInt(x) => x.shape_len() * std::mem::size_of::<u32>(),            
            TensorHolder::Float(x) => x.shape_len() * std::mem::size_of::<f32>(),            
            TensorHolder::Double(x) => x.shape_len() * std::mem::size_of::<f64>(),            
        }) as wgpu::BufferAddress
    }

    pub(crate) fn is_single(&self) -> bool {
        match self {
            TensorHolder::Int(x) => x.is_single(),
            TensorHolder::UInt(x) => x.is_single(),
            TensorHolder::Float(x) => x.is_single(),
            TensorHolder::Double(x) => x.is_single()
        }
    }

    pub(crate) fn get_type_glsl(&self) -> String {
        match self {
            TensorHolder::Int(x) => if x.is_single() {"int"} else {"int[]"},
            TensorHolder::UInt(x) => if x.is_single() {"uint"} else {"uint[]"},
            TensorHolder::Float(x) => if x.is_single() {"float"} else {"float[]"},
            TensorHolder::Double(x) => if x.is_single() {"double"} else {"double[]"},
        }.to_string()
    }

    pub(crate) fn copy(&self) -> Self {
        match self {
            TensorHolder::Int(x) => TensorHolder::Int(x),
            TensorHolder::UInt(x) => TensorHolder::UInt(x),
            TensorHolder::Float(x) => TensorHolder::Float(x),
            TensorHolder::Double(x) => TensorHolder::Double(x),
        }
    }
}

impl<'a> PartialEq for TensorHolder<'a> {
    fn eq(&self, other: &Self) -> bool {
        match self {
            TensorHolder::Int(x) => {
                match other {
                    TensorHolder::Int(y) => ptr::eq(*x, *y),
                    _ => false
                }
            }
            TensorHolder::UInt(x) => {
                match other {
                    TensorHolder::UInt(y) => ptr::eq(*x, *y),
                    _ => false
                }
            }
            TensorHolder::Float(x) => {
                match other {
                    TensorHolder::Float(y) => ptr::eq(*x, *y),
                    _ => false
                }
            }
            TensorHolder::Double(x) => {
                match other {
                    TensorHolder::Double(y) => ptr::eq(*x, *y),
                    _ => false
                }
            }
        }
    }
}