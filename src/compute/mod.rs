pub mod gpu;

pub fn run() -> Vec<u32> {
    gpu::build_shader();
    //gpu::public_run()
    Vec::new()
}