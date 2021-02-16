# gda_compute
## Project Status
A very early version of the project (like 0.0.1 pre-alpha early). It is by no means usable by now and everything is under evaluation. That said, if you still want to try it out, 
beware that at this stage everything could, and probably should, change.
Also: There are a lot of compile warnings for now, this is mostly because I have not finalized my decisions on Error handling. I will probably 
pass most errors to the caller, but for now, there is a lot of panic potential.

## What it is
This projects goal is to provide an easy to use and clean API for computation on GPU and CPU. It uses wgpu-rs for the gpu execution and currently builds shaders JIT.
For now the focus is on Tensor computation, but as I 
stated before, this is not final. It might be odd that I started a project without a specific goal in mind. The reason for this is, that I
started the project mainly as a learning project for myself. So maybe it will not go anywhere stable, but we will see.

## How to use it
For now: don't! But if you still want to try it out, there are some "examples" in the test cases located at /compute/mod.rs .

You still want an example? Here you go:
```rust
let a = Tensor::new(vec![1f32]); // create a Scalar
let b = Tensor::with_shape(vec![1f32,1.0,1.0], vec![3]); // 1D Vector 
let c = Tensor::new(vec![1f32,1.0,1.0]); // also a 1D Vector

let op = &a + &b -&c;    
let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(op.get_shape()); /* just important for the result type, the shape could actually be infered */

let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
let shader = gpu.build(op, result_tensor); //build only once 

let result = gpu.execute(&shader).unwrap(); // result = [1.0,1.0,1.0]

a.set(&[0], 2f32); //change single values
b.change_value(vec![5f32,4.0,3.0]); /* change the whole value, no shape changing for now */

let result2 = gpu.execute(&shader).unwrap(); // result2 = [6.0, 5.0, 4.0]
```

Note: If you want to use the project you currently need an installation of shaderc on your system. In the future I want to get rid of this dependency,
either by building spirv directly or more likely building WGSL as soon as it is somewhat stable in wgpu. 
More information on shaderc is provided here: https://github.com/google/shaderc

## Todos
- Define a project goal
- Error Handling, no panic
- Basic functionality
- CPU backend
- Test cases
- A lot more

## Have some thoughts/ want to Contribute?
Feel free to open an issue or a pull request. Every opinion and help is welcome!
