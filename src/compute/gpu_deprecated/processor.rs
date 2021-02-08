use std::fmt::Write;
use std::cell::RefCell;
use std::cell::Ref;
use std::rc::Rc;
use std::ops::{Add};
use std::{convert::TryInto, str::FromStr, option};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup, BindGroupLayout};
use wgpu;

pub struct TensorBinding<'a> {
    id: u32,
    value: &'a Tensor,
    change: Rc<RefCell<u32>>
}

impl<'a> TensorBinding<'a> {
    pub fn has_changed(&self) -> bool {
        if self.value.get_change() > *self.change.borrow() {
            *self.change.borrow_mut() = self.value.get_change();
            return true
        }
        false
    }
}

pub struct Tensor {
    value: RefCell<Vec<f32>>,
    change: Rc<RefCell<u32>>,
    //shape: Vec<u32>
}

impl Tensor {
    pub fn new(vec: Vec<f32>) -> Tensor {
        Tensor {value: RefCell::new(vec), change: Rc::new(RefCell::new(0))}
    }

    pub fn get_value(&self) -> Ref<Vec<f32>> {
        self.value.borrow()
    }

    pub fn set_value(&self, mut val: Vec<f32>) {
        let mut old_val = self.value.borrow_mut();
        old_val.clear();
        old_val.append(&mut val);
        *self.change.borrow_mut() += 1;
    }

    pub fn get_change(&self) -> u32 {
        *self.change.borrow()
    }
}

impl<'a> Add for &'a Tensor {
    type Output = Operation<'a>;

    fn add(self, other: Self) -> Operation<'a> {
        Operation::Add(Box::new(Operation::Var(
            Box::new(TensorBinding {id: 0, value: self, change: Rc::new(RefCell::new(self.get_change()))}))), 
            Box::new(Operation::Var(Box::new(TensorBinding {id: 1, value: other, change: Rc::new(RefCell::new(other.get_change()))}))))
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
            writeln!(&mut s, "float[] indices{};", (*i).id);
            writeln!(&mut s, "}};");
            writeln!(&mut s);
            if binding < (*i).id {
                binding = (*i).id+1;
            }
        }

        writeln!(&mut s, "layout(set = 0, binding = {}) buffer b{} {{", binding, binding);
        writeln!(&mut s, "float[] result;");
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
        "result[index] = indices0[index] * indices1[index];"
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

pub struct Shader<'a> {
    spirv: Vec<u8>,
    result_binding: u32,
    inputs: Vec<&'a TensorBinding<'a>>,
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    staging_buffer: Buffer,
    storage_buffers: Vec<Buffer>,
    result_buffer: Buffer,
    result_size: wgpu::BufferAddress,
}

impl<'a> Shader<'a> {
    pub fn build(op: &'a Operation<'a>, gpu: &mut super::GPU) -> Shader<'a> { 
        if let Some(device) = gpu.device.as_mut() {
            let (spirv, inputs, result_binding) = op.build(); 

            inputs.iter().for_each(|x| drop(x.has_changed()));

            let cs_module = (*device).create_shader_module(wgpu::util::make_spirv(&spirv));

            let resurlt_vec = vec![0f32,0.0,0.0];
            let slice_size = resurlt_vec.len() * std::mem::size_of::<u32>();
            let size = slice_size as wgpu::BufferAddress;

            let staging_buffer = (*device).create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            let storage_buffers: Vec<wgpu::Buffer> = inputs.iter().map(|&i| {
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

            let mut layout: Vec<wgpu::BindGroupLayoutEntry> = inputs.iter().map(|&i| { 
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
                binding: result_binding,
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

            for c in 0..inputs.len() {
                b_group.push(wgpu::BindGroupEntry {
                    binding: inputs[c].id,
                    resource: wgpu::BindingResource::Buffer(storage_buffers[c].slice(..)),
                });
            }

        
            b_group.push(wgpu::BindGroupEntry {
                binding: result_binding,
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
            
            return Shader {spirv, result_binding, inputs, staging_buffer, compute_pipeline, storage_buffers, bind_group, bind_group_layout, result_buffer, result_size: size}
        }
        panic!("No GPU!");
    }

    pub fn execute(&mut self, gpu: &mut super::GPU) -> Vec<f32> {
        if let Some(device) = gpu.device.as_mut() {            
            let mut changed = false;
            for i in 0..self.inputs.len() {
                if self.inputs[i].has_changed() {
                    changed = true;
                    self.storage_buffers[i] = (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("Storage Buffer"),
                        contents: bytemuck::cast_slice(&self.inputs[i].value.get_value()),
                        usage: wgpu::BufferUsage::STORAGE
                            | wgpu::BufferUsage::COPY_DST
                            | wgpu::BufferUsage::COPY_SRC,
                    });                  
                }
            }

            if changed {
                let mut b_group = Vec::<wgpu::BindGroupEntry>::new();

                for c in 0..self.inputs.len() {
                    b_group.push(wgpu::BindGroupEntry {
                        binding: self.inputs[c].id,
                        resource: wgpu::BindingResource::Buffer(self.storage_buffers[c].slice(..)),
                    });
                }

            
                b_group.push(wgpu::BindGroupEntry {
                    binding: self.result_binding,
                    resource: wgpu::BindingResource::Buffer(self.result_buffer.slice(..)),
                });

                self.bind_group = (*device).create_bind_group(&wgpu::BindGroupDescriptor {
                    label: None,
                    layout: &self.bind_group_layout,
                    entries: &b_group,
                });
            }

            let queue = &mut gpu.queue.as_mut().unwrap();
            let mut encoder =
                (*device).create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
            {
                let mut cpass = encoder.begin_compute_pass();
                cpass.set_pipeline(&self.compute_pipeline);
                cpass.set_bind_group(0, &self.bind_group, &[]);
                cpass.insert_debug_marker("compute collatz iterations");
                cpass.dispatch(self.result_size as u32, 1, 1);
            }
            encoder.copy_buffer_to_buffer(&self.result_buffer, 0, &self.staging_buffer, 0, self.result_size);
        
            (*queue).submit(Some(encoder.finish()));
        
            // Note that we're not calling `.await` here.
            let buffer_slice = self.staging_buffer.slice(..);
            let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);
        
            // Poll the device in a blocking manner so that our future resolves.
            // In an actual application, `device.poll(...)` should
            // be called in an event loop or on another thread.
            (*device).poll(wgpu::Maintain::Wait);
        
            if let Ok(()) = pollster::block_on(buffer_future) {
                let data = buffer_slice.get_mapped_range();
                let result = data
                    .chunks_exact(4)
                    .map(|b| f32::from_ne_bytes(b.try_into().unwrap()))
                    .collect();
            
                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                drop(data);
                self.staging_buffer.unmap();
            
                return result
            } else {
                panic!("failed to run compute on gpu!")
            }
        }
        panic!("failed to run compute on gpu!")
    }
}