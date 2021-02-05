use std::{convert::TryInto, str::FromStr, option};
use wgpu::util::DeviceExt;
use wgpu::{Device, Queue, Buffer, ComputePipeline, BindGroup};
use wgpu;

pub mod processor;
use processor::*;

pub fn build_shader() {
    let mut a = Tensor::new(vec![1,1,1]);
    let mut b = Tensor::new(vec![1,1,1]);
    let op = &a + &b;
    let shader = Shader::build(&op);
    let mut gpu = GPU::new();
    let res1 = shader.execute(&mut gpu);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res1);

    b.set_value(vec![4,4,4]);
    let res2 = shader.execute(&mut gpu);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res2);
}

trait GPUOperation<TInput, TOutput> {
    fn run(&mut self, gpu: &mut GPU) -> TOutput;
}

struct TestOp {
    size: wgpu::BufferAddress,
    input: Vec<u32>,
    compute_pipeline: ComputePipeline,
    bind_group: BindGroup,
    staging_buffer: Buffer,
    storage_buffer: Buffer
}

impl TestOp {
    fn new(input: Vec<u32>, gpu: &mut GPU) -> Option<TestOp> {
        let slice_size = input.len() * std::mem::size_of::<u32>();
        let size = slice_size as wgpu::BufferAddress;

        if let Some(device) = gpu.device.as_mut() {
            let cs_module = (*device).create_shader_module(wgpu::include_spirv!("shaders\\test.comp.spv"));

            let staging_buffer = (*device).create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });

            let storage_buffer = (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });

            let bind_group_layout = (*device).create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                }],
            });

            let bind_group = (*device).create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
                }],
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

            return Some(TestOp {
                size,
                input,
                compute_pipeline,
                bind_group,
                storage_buffer,
                staging_buffer
            })
        }    
        None    
    }
}

impl GPUOperation<Vec<u32>, Vec<u32>> for TestOp {
    fn run(&mut self, gpu: &mut GPU) -> Vec<u32> {
        let device = &mut gpu.device.as_mut().unwrap();
        let queue = &mut gpu.queue.as_mut().unwrap();

        let mut encoder =
            (*device).create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        {
            let mut cpass = encoder.begin_compute_pass();
            cpass.set_pipeline(&self.compute_pipeline);
            cpass.set_bind_group(0, &self.bind_group, &[]);
            cpass.insert_debug_marker("compute collatz iterations");
            cpass.dispatch(self.input.len() as u32, 1, 1);
        }
        encoder.copy_buffer_to_buffer(&self.storage_buffer, 0, &self.staging_buffer, 0, self.size);

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
                .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
                .collect();

            // With the current interface, we have to make sure all mapped views are
            // dropped before we unmap the buffer.
            drop(data);
            self.staging_buffer.unmap();

            result
        } else {
            panic!("failed to run compute on gpu!")
        }
    }

    
}

pub struct GPU {
    exist: bool,
    device: Option<Device>,
    queue: Option<Queue>
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

 /*   fn GetOperation<T>(&mut self, inputs: Vec<Vec<T>>, output: Vec<T>, shader: String) {
        Vec sizes = Vec<wgpu::BufferAddress>::new();
        for(let i in inputs) {
            let slice_size = input.len() * std::mem::size_of::<u32>();
            let size = slice_size as wgpu::BufferAddress;
            sizes.push(size);
        }

        if let Some(device) = gpu.device.as_mut() {
            let file = fs::read(Path::from(shader));

            let cs_module = (*device).create_shader_module(wgpu::util::make_spirv(&file));

            let staging_buffer = (*device).create_buffer(&wgpu::BufferDescriptor {
                label: None,
                size,
                usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
                mapped_at_creation: false,
            });
            
            for(let i in [0..inputs.len()+1) {

            }
            let storage_buffer = (*device).create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Storage Buffer"),
                contents: bytemuck::cast_slice(&input),
                usage: wgpu::BufferUsage::STORAGE
                    | wgpu::BufferUsage::COPY_DST
                    | wgpu::BufferUsage::COPY_SRC,
            });

            let bind_group_layout = (*device).create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStage::COMPUTE,
                    ty: wgpu::BindingType::StorageBuffer {
                        dynamic: false,
                        readonly: false,
                        min_binding_size: wgpu::BufferSize::new(4),
                    },
                    count: None,
                }],
            });

            let bind_group = (*device).create_bind_group(&wgpu::BindGroupDescriptor {
                label: None,
                layout: &bind_group_layout,
                entries: &[wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
                }],
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
        }    
    } */
}

async fn run() {
    let numbers = if std::env::args().len() <= 1 {
        let default = vec![1, 2, 3, 4];
        println!("No numbers were provided, defaulting to {:?}", default);
        default
    } else {
        std::env::args()
            .skip(1)
            .map(|s| u32::from_str(&s).expect("You must pass a list of positive integers!"))
            .collect()
    };

    let times = execute_gpu(numbers).await;
    println!("Times: {:?}", times);
    #[cfg(target_arch = "wasm32")]
    log::info!("Times: {:?}", times);
}

async fn get_gpu_device() {
    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    
}

async fn execute_gpu(numbers: Vec<u32>) -> Vec<u32> {
    let slice_size = numbers.len() * std::mem::size_of::<u32>();
    let size = slice_size as wgpu::BufferAddress;

    let instance = wgpu::Instance::new(wgpu::BackendBit::PRIMARY);
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::Default,
            compatible_surface: None,
        })
        .await
        .unwrap();

    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                features: wgpu::Features::empty(),
                limits: wgpu::Limits::default(),
                shader_validation: true,
            },
            None,
        )
        .await
        .unwrap();

    let cs_module = device.create_shader_module(wgpu::include_spirv!("shaders\\test.comp.spv"));

    let staging_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size,
        usage: wgpu::BufferUsage::MAP_READ | wgpu::BufferUsage::COPY_DST,
        mapped_at_creation: false,
    });

    let storage_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
        label: Some("Storage Buffer"),
        contents: bytemuck::cast_slice(&numbers),
        usage: wgpu::BufferUsage::STORAGE
            | wgpu::BufferUsage::COPY_DST
            | wgpu::BufferUsage::COPY_SRC,
    });

    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: None,
        entries: &[wgpu::BindGroupLayoutEntry {
            binding: 0,
            visibility: wgpu::ShaderStage::COMPUTE,
            ty: wgpu::BindingType::StorageBuffer {
                dynamic: false,
                readonly: false,
                min_binding_size: wgpu::BufferSize::new(4),
            },
            count: None,
        }],
    });

    let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
        label: None,
        layout: &bind_group_layout,
        entries: &[wgpu::BindGroupEntry {
            binding: 0,
            resource: wgpu::BindingResource::Buffer(storage_buffer.slice(..)),
        }],
    });

    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: None,
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });

    let compute_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: None,
        layout: Some(&pipeline_layout),
        compute_stage: wgpu::ProgrammableStageDescriptor {
            module: &cs_module,
            entry_point: "main",
        },
    });

    let mut encoder =
        device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
    {
        let mut cpass = encoder.begin_compute_pass();
        cpass.set_pipeline(&compute_pipeline);
        cpass.set_bind_group(0, &bind_group, &[]);
        cpass.insert_debug_marker("compute collatz iterations");
        cpass.dispatch(numbers.len() as u32, 1, 1);
    }
    encoder.copy_buffer_to_buffer(&storage_buffer, 0, &staging_buffer, 0, size);

    queue.submit(Some(encoder.finish()));

    // Note that we're not calling `.await` here.
    let buffer_slice = staging_buffer.slice(..);
    let buffer_future = buffer_slice.map_async(wgpu::MapMode::Read);

    // Poll the device in a blocking manner so that our future resolves.
    // In an actual application, `device.poll(...)` should
    // be called in an event loop or on another thread.
    device.poll(wgpu::Maintain::Wait);

    if let Ok(()) = buffer_future.await {
        let data = buffer_slice.get_mapped_range();
        let result = data
            .chunks_exact(4)
            .map(|b| u32::from_ne_bytes(b.try_into().unwrap()))
            .collect();

        // With the current interface, we have to make sure all mapped views are
        // dropped before we unmap the buffer.
        drop(data);
        staging_buffer.unmap();

        result
    } else {
        panic!("failed to run compute on gpu!")
    }
}

pub fn public_run() -> Vec<u32> {
    #[cfg(not(target_arch = "wasm32"))]
    {
        wgpu_subscriber::initialize_default_subscriber(None);
        let input = vec![5, 23, 10, 9];
        //let result = pollster::block_on(execute_gpu(input));// expected result: vec![5, 15, 6, 19]
        let mut gpu = GPU::new();
        let mut op = TestOp::new(input, &mut gpu).unwrap();
        let result = op.run(&mut gpu);
        result
    }
    #[cfg(target_arch = "wasm32")]
    {
        std::panic::set_hook(Box::new(console_error_panic_hook::hook));
        console_log::init().expect("could not initialize logger");
        wasm_bindgen_futures::spawn_local(run());
        Vec::<u32>::new()
    }    
}

#[cfg(all(test, not(target_arch = "wasm32")))]
mod tests {
    use super::*;

    #[test]
    fn test_compute_1() {
        let input = vec![1, 2, 3, 4];
        pollster::block_on(assert_execute_gpu(input, vec![0, 1, 7, 2]));
    }

    #[test]
    fn test_compute_2() {
        let input = vec![5, 23, 10, 9];
        pollster::block_on(assert_execute_gpu(input, vec![5, 15, 6, 19]));
    }

    #[test]
    fn test_multithreaded_compute() {
        use std::{sync::mpsc, thread, time::Duration};

        let thread_count = 8;

        let (tx, rx) = mpsc::channel();
        for _ in 0..thread_count {
            let tx = tx.clone();
            thread::spawn(move || {
                let input = vec![100, 100, 100];
                pollster::block_on(assert_execute_gpu(input, vec![25, 25, 25]));
                tx.send(true).unwrap();
            });
        }

        for _ in 0..thread_count {
            rx.recv_timeout(Duration::from_secs(10))
                .expect("A thread never completed.");
        }
    }

    async fn assert_execute_gpu(input: Vec<u32>, expected: Vec<u32>) {
        assert_eq!(execute_gpu(input).await, expected);
    }
}