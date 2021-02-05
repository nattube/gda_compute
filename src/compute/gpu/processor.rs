use std::fmt::Write;
use std::cell::RefCell;
use std::cell::Ref;
use std::ops::{Add};
use std::{convert::TryInto, str::FromStr, option};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup};
use wgpu;

pub struct TensorBinding<'a> {
    id: u32,
    value: &'a Tensor
}

pub struct Tensor {
    value: RefCell<Vec<u32>>,
    shape: Vec<u32>
}

impl Tensor {
    pub fn new(vec: Vec<u32>) -> Tensor {
        Tensor {value: RefCell::new(vec)}
    }

    pub fn get_value(&self) -> Ref<Vec<u32>> {
        self.value.borrow()
    }

    pub fn set_value(&self, mut val: Vec<u32>) {
        let mut old_val = self.value.borrow_mut();
        old_val.clear();
        old_val.append(&mut val);
    }
}

impl<'a> Add for &'a Tensor {
    type Output = Operation<'a>;

    fn add(self, other: Self) -> Operation<'a> {
        Operation::Add(Box::new(Operation::Var(Box::new(TensorBinding {id: 0, value: self}))), Box::new(Operation::Var(Box::new(TensorBinding {id: 1, value: other}))))
    }
}

pub enum Operation<'a> {    
    Var(Box<TensorBinding<'a>>),
    Add(Box<Operation<'a>>, Box<Operation<'a>>),
    Mult(Box<Operation<'a>>, Box<Operation<'a>>)
}

impl<'a> Operation<'a> {
    
}

impl<'a> Operation<'a> {
    fn build(&'a self) -> (Vec<u8>, Vec<&'a TensorBinding<'a>>, u32) {        
        let mut inputs = Vec::<&'a TensorBinding<'a>>::new();
        self.fill_input(&mut inputs);
        
        let mut s = String::new();
        writeln!(&mut s, "#version 450");
        writeln!(&mut s, "layout(local_size_x = 1) in;");
        
        writeln!(&mut s);
        
        let mut binding: u32 = 0;

        for i in &inputs {
            writeln!(&mut s, "readonly layout(set = 0, binding = {}) buffer b{} {{", (*i).id, (*i).id);
            writeln!(&mut s, "uint[] indices{};", (*i).id);
            writeln!(&mut s, "}};");
            writeln!(&mut s);
            if binding < (*i).id {
                binding = (*i).id+1;
            }
        }

        writeln!(&mut s, "layout(set = 0, binding = {}) buffer b{} {{", binding, binding);
        writeln!(&mut s, "uint[] result;");
        writeln!(&mut s, "}};");
        writeln!(&mut s);
        
        writeln!(&mut s, "void main() {{");
        writeln!(&mut s, "uint index = gl_GlobalInvocationID.x;");
        writeln!(&mut s, "{}", self.build_str());
        writeln!(&mut s, "}}");

        println!("{}", s);

        (Operation::build_shader(&s), inputs, binding)
    }

    fn build_str(&self) -> &str {
        "result[index] = indices0[index] + indices1[index];"
    }

    fn fill_input(&'a self, inputs: &mut Vec<&'a TensorBinding<'a>>) {
        match self {
            Operation::Var(x) => {
                if !inputs.iter().any(|&y| y.id == x.id) {
                    inputs.push(&x);
                }
            },
            Operation::Add(x, y) | Operation::Mult(x, y) => {
                x.fill_input(inputs);
                y.fill_input(inputs);
            }
        }
    }

    fn build_shader(src: &str) -> Vec<u8> {
        let mut compiler = shaderc::Compiler::new().unwrap();
        let binary_result = compiler.compile_into_spirv(
            src, shaderc::ShaderKind::Compute,
            "shader.glsl", "main", None).unwrap();

        binary_result.as_binary_u8().to_owned()
    }
}

trait AbstractProcessor {
    fn add();
}

pub struct Shader<'a> {
    spirv: Vec<u8>,
    result_binding: u32,
    inputs: Vec<&'a TensorBinding<'a>>
}

impl<'a> Shader<'a> {
    pub fn build(op: &'a Operation<'a>) -> Shader<'a> {
        let (spirv, inputs, result_binding) = op.build();        

        return Shader {spirv, result_binding, inputs}
    }

    pub fn execute(&self, gpu: &mut super::GPU) -> Vec<u32> {
        if let Some(device) = gpu.device.as_mut() {
            let queue = &mut gpu.queue.as_mut().unwrap();
            let cs_module = (*device).create_shader_module(wgpu::util::make_spirv(&self.spirv));

            let resurlt_vec = vec![0,0,0];
            let slice_size = resurlt_vec.len() * std::mem::size_of::<u32>();
            let size = slice_size as wgpu::BufferAddress;

            let staging_buffer = (*device).create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            let storage_buffers: Vec<wgpu::Buffer> = self.inputs.iter().map(|&i| {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&i.value.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })
            }).collect();

            

            let result_buffer = (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Result Buffer"),
                contents: bytemuck::cast_slice(&resurlt_vec),
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });

            let mut layout: Vec<wgpu::BindGroupLayoutEntry> = self.inputs.iter().map(|&i| { 
                wgpu::BindGroupLayoutEntry {
                    binding: i.id,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: wgpu::BufferSize::new(3),
                    },
                    count: None,
                }
            }).collect();

            layout.push(wgpu::BindGroupLayoutEntry {
                binding: self.result_binding,
                visibility: wgpu::ShaderStage::COMPUTE,
                ty: wgpu::BindingType::StorageBuffer {
                    dynamic: false,
                    readonly: false,
                    min_binding_size: wgpu::BufferSize::new(3),
                },
                count: None,
            });

            let bind_group_layout = (*device).create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &layout,
            });

            let mut b_group = Vec::<wgpu::BindGroupEntry>::new();

            for c in 0..self.inputs.len() {
                b_group.push(wgpu::BindGroupEntry {
                    binding: self.inputs[c].id,
                    resource: wgpu::BindingResource::Buffer(storage_buffers[c].slice(..)),
                });
            }

        
            b_group.push(wgpu::BindGroupEntry {
                binding: self.result_binding,
                resource: wgpu::BindingResource::Buffer(result_buffer.slice(..)),
            });

            let bind_group = (*device).create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &b_group,
            });

            
            let pipeline_layout = (*device).create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bind_group_layout],
                push_constant_ranges: &[],
            });

            let compute_pipeline = (*device).create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&pipeline_layout),
                compute_stage: wgpu::ProgrammableStageDescriptor {
                    module: &cs_module,
                    entry_point: "main",
                },
            });

            let mut encoder =
                (*device).create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass();
                cpass.set_pipeline(&compute_pipeline);
                cpass.set_bind_group(0, &bind_group, &[]);
                cpass.insert_debug_marker("compute collatz iterations");
                cpass.dispatch(size as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&result_buffer, 0, &staging_buffer, 0, size);
        
            (*queue).submit(Some(encoder.finish()));
        
            // Note that we're not calling `.await` here.
            let buffer_slice = staging_buffer.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        
            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            (*device).poll(wgpu::Maintain::Wait);
        
            if let Ok(()) = pollster::block_on(buffer_future) {
                let data = buffer_slice.get_mapped_range();
                let result = data
                    .chunks_exact(4)
                    .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
            
                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                drop(data);
                staging_buffer.unmap();
            
                return result
            } else {
                panic!("failed to run compute on gpu!")
            }
        }
        panic!("failed to run compute on gpu!")
    }
}