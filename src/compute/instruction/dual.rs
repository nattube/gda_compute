pub trait DualInstruction {
    pub fn build_gpu(left: String, right: String) -> String;
    pub fn build_cpu<T>(left: T, right: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T>;
}


pub struct AddInstruction {}
pub struct SubtInstruction {}
pub struct MultInstruction {}
pub struct DivInstruction {}

impl DualInstruction for AddInstruction {
    pub fn build_gpu(left: String, right: String) -> String {
        let mut res = String::new();
        write!(&mut res, "{} + {}", left, right);
        res
    }

    pub fn build_cpu<T>(left: T, right: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> 
    {
        left + right
    }
}

impl DualInstruction for SubtInstruction {
    pub fn build_gpu(left: String, right: String) -> String {
        let mut res = String::new();
        write!(&mut res, "{} - {}", left, right);
        res
    }

    pub fn build_cpu<T>(left: T, right: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> 
    {
        left - right
    }
}

impl DualInstruction for MultInstruction {
    pub fn build_gpu(left: String, right: String) -> String {
        let mut res = String::new();
        write!(&mut res, "{} * {}", left, right);
        res
    }

    pub fn build_cpu<T>(left: T, right: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> 
    {
        left * right
    }
}

impl DualInstruction for DivInstruction {
    pub fn build_gpu(left: String, right: String) -> String {
        let mut res = String::new();
        write!(&mut res, "{} / {}", left, right);
        res
    }

    pub fn build_cpu<T>(left: T, right: T) -> T
        where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> 
    {
        left / right
    }
}