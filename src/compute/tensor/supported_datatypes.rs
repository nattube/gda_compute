use super::{SupportedDataTypes, Tensor, TensorHolder};

impl SupportedDataTypes for i32 {
    type BindingType = i32;
    fn to_data_holder(vec: &Tensor<Self::BindingType>) -> TensorHolder {
        TensorHolder::Int(vec)
    }
    fn strength() -> usize {
        0
    }
    fn get_zero() -> Self::BindingType {
        0i32
    }
}

impl SupportedDataTypes for u32 {
    type BindingType = u32;
    fn to_data_holder(vec: &Tensor<Self::BindingType>) -> TensorHolder {
        TensorHolder::UInt(vec)
    }
    fn strength() -> usize {
        1
    }
    fn get_zero() -> Self::BindingType {
        0u32
    }
}

impl SupportedDataTypes for f32 {
    type BindingType = f32;
    fn to_data_holder(vec: &Tensor<Self::BindingType>) -> TensorHolder {
        TensorHolder::Float(vec)
    }
    fn strength() -> usize {
        2
    }
    fn get_zero() -> Self::BindingType {
        0f32
    }
}
impl SupportedDataTypes for f64 {
    type BindingType = f64;
    fn to_data_holder(vec: &Tensor<Self::BindingType>) -> TensorHolder {
        TensorHolder::Double(vec)
    }
    fn strength() -> usize {
        3
    }
    fn get_zero() -> Self::BindingType {
        0f64
    }
}