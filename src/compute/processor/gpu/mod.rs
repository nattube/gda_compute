use wgpu::{Device, Queue};
use wgpu;

use std::cell::RefCell;
use std::rc::Rc;

pub mod shader;
use shader::Shader;
use crate::compute::tensor::{Operation};
use crate::compute::processor::Compiled;

pub struct GPU {
    pub(crate) exist: bool,
    pub(crate) device: Option<Device>,
    pub(crate) queue: Option<Queue>
}

impl<'a> super::AbstractProcessor<'a> for GPU {
    fn build(&mut self, op: &'a Operation<'a>) -> Compiled<'a> {
        Compiled::GPU(Rc::new(RefCell::new(Shader::build(op, self))))
    }

    fn execute(&mut self, compiled: &Compiled<'a>) -> Vec<f32> {
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

        if let Ok((device, queue)) = 
            pollster::block_on(adapter
                .request_device(
                    &wgpu::DeviceDescriptor {
                        features: wgpu::Features::empty(),
                        limits: wgpu::Limits::default(),
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