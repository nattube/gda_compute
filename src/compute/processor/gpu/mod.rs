use wgpu::{Device, Queue};
use wgpu;

use std::cell::RefCell;
use std::rc::Rc;

pub mod shader;
use shader::Shader;
use crate::compute::tensor::{Operation, SupportedDataTypes, Tensor, TensorOperationResult};
use crate::compute::processor::Compiled;

pub struct GPU {
    pub(crate) exist: bool,
    pub(crate) device: Option<Device>,
    pub(crate) queue: Option<Queue>
}

impl GPU {
    pub(crate) fn build<'a, T>(&mut self, op: Operation<'a>, tensor: Tensor<T>) -> Compiled<'a, T> 
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        Compiled::GPU(Rc::new(RefCell::new(Shader::build(op, self, tensor))))
    }

    pub(crate) fn execute<'a, T>(&mut self, compiled: &Compiled<'a, T>) -> TensorOperationResult 
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        return match compiled {
            Compiled::GPU(c) => (*c.borrow_mut()).execute(self)
        }
    }
}

impl GPU {
    pub(crate) fn new() -> GPU {
        let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
        let adapter = 
            pollster::block_on(instance
                .request_adapter(&wgpu::RequestAdapterOptions {
                    power_preference: wgpu::PowerPreference::Default,
                    compatible_surface: None,
        })).unwrap();

        //println!("{:?}", adapter.limits());

        if let Ok((device, queue)) = 
            pollster::block_on(adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: wgpu::Features::empty(),
                        limits: adapter.limits(),
                        shader_validation: true,
                    },
                    None,
        )) {
            return GPU {
                exist: true,
                device: Some(device),
                queue: Some(queue)
            }
        }

        GPU {
            exist: false,
            device: None,
            queue: None
        }
    }
}