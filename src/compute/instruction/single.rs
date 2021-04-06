pub trait SingleInstruction {
    pub fn build_gpu(value: String) -> String;
    pub fn build_cpu<T>(value: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T>;
}

pub struct SqrtInstruction {}

impl SingleInstruction for SqrtInstruction {
    pub fn build_gpu(value: String) -> String {
        let mut res = String::new();
        write!(&mut res, "sqrt({})", value);
        res
    }

    pub fn build_cpu<T>(value: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> 
    {
        (values as f64).sqrt() as T
    }
} 