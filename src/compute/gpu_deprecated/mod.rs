use std::{convert::TryInto, str::FromStr, option};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup};
use wgpu;

pub mod processor;
use processor::*;

pub fn build_shader() {
    let mut a = Tensor::new(vec![1f32,1.0,1.0]);
    let mut b = Tensor::new(vec![1f32,1.0,1.0]);
    let op = &a + &b;
    let mut gpu = GPU::new();
    let mut shader = Shader::build(&op, &mut gpu);
    let res1 = shader.execute(&mut gpu);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res1);

    b.set_value(vec![4f32,3.0,2.0]);
    let res2 = shader.execute(&mut gpu);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res2);
}

pub struct GPU {
    exist: bool,
    device: Option<Device>,
    queue: Option<Queue>
}

impl super::AbstractProcessor for GPU {
    
}

impl GPU {
    pub fn new() -> GPU {
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