use std::{convert::TryInto};
use wgpu::util::DeviceExt;
use wgpu::{Buffer, ComputePipeline, BindGroup, BindGroupLayout, Device};
use wgpu;
use crate::compute::tensor::{Operation, Tensor, TensorBinding, TensorError, TensorOperationResult, TensorHolder};

pub struct Shader<'a> {
    spirv: Vec<u8>,
    result_binding: u32,
    inputs: Vec<TensorBinding<'a>>,
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    bind_group_layout: BindGroupLayout,
    staging_buffer: Buffer,
    storage_buffers: Vec<Buffer>,
    result_buffer: Buffer,
    result_size: wgpu::BufferAddress,
    tensor_result: TensorOperationResult
}

impl<'a> Shader<'a> {
    pub(crate) fn build(op: Operation<'a>, gpu: &mut super::GPU) -> Shader<'a> { 
        if let Some(device) = gpu.device.as_mut() {            
            let (spirv, inputs, result_binding) = op.build_gpu(); 

            inputs.iter().for_each(|x| drop(x.has_changed()));

            let cs_module = (*device).create_shader_module(wgpu::util::make_spirv(&spirv));

            let tensor_result = match op {
                Operation::DualOp {result, ..} | Operation::SingleOp {result, ..} => result.copy(),
                _ => panic!("unimplemented error case")
            };

            if let TensorOperationResult::Error(x) = tensor_result {
                panic!("unimplemented error case")
            }

            println!("Result vec form: {:?}", tensor_result);

            let size = tensor_result.get_mem_size();

            let staging_buffer = (*device).create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            let storage_buffers: Vec<wgpu::Buffer> = inputs.iter().map(|i| {
                Shader::get_buffer(device, &i.value, "Storage Buffer")
            }).collect();            

            let result_buffer = Self::get_buffer_from_opres(device, &tensor_result, "Result Buffer");

            let mut layout: Vec<wgpu::BindGroupLayoutEntry> = inputs.iter().map(|i| { 
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
            
            return Shader {spirv, result_binding, inputs, staging_buffer, compute_pipeline, storage_buffers, bind_group, bind_group_layout, result_buffer, result_size: size, tensor_result}
        }
        panic!("No GPU!");
    }

    fn get_buffer_from_opres(device: &mut Device, tensor: &TensorOperationResult, label: &str) -> wgpu::Buffer {
        return match tensor {
            TensorOperationResult::Int(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorOperationResult::UInt(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorOperationResult::Float(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorOperationResult::Double(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            _ => panic!("Unimlemented Error handling")
        }
    }

    fn get_buffer(device: &mut Device, tensor: &TensorHolder, label: &str) -> wgpu::Buffer {
        return match tensor {
            TensorHolder::Int(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorHolder::UInt(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorHolder::Float(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            TensorHolder::Double(x) => {
                (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("Storage Buffer"),
                    contents: bytemuck::cast_slice(&x.get_value()),
                    usage: wgpu::BufferUsage::STORAGE
                        | wgpu::BufferUsage::COPY_DST
                        | wgpu::BufferUsage::COPY_SRC,
                })              
            },
            _ => panic!("Unimlemented Error handling")
        }
    }

    pub(crate) fn execute(&mut self, gpu: &mut super::GPU) -> TensorOperationResult {
        if let Some(device) = gpu.device.as_mut() {            
            let mut changed = false;
            for i in 0..self.inputs.len() {
                if self.inputs[i].has_changed() {
                    changed = true;
                    self.storage_buffers[i] = Shader::get_buffer(device, &self.inputs[i].value, "Storage Buffer");
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

                //TODO: result_buffers shape shoud be implemented as relationship to other
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
                self.tensor_result.map_from_staging_buffer(data);
            
                // With the current interface, we have to make sure all mapped views are
                // dropped before we unmap the buffer.
                self.staging_buffer.unmap();
            
                return self.tensor_result.copy()
            } else {
                return TensorOperationResult::Error(TensorError::Unimplemented("failed to run compute on gpu!".to_owned()))
            }
        }
        return TensorOperationResult::Error(TensorError::Unimplemented("failed to run compute on gpu!".to_owned()))
    }
}