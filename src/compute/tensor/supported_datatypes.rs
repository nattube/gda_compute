use super::{SupportedDataTypes, Shape, Tensor, TensorError, TensorHolder, TensorOperationResult};

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
    fn get_tensor(res: TensorOperationResult, wanted_shape: &Shape) -> Result<Tensor<Self::BindingType>, TensorError> {
        match res {
            TensorOperationResult::Int(x) => {
                if !x.matches_shape(wanted_shape) {return Err(TensorError::ShapeError("Result Shapes didn't match".to_owned()))}
                Ok(Tensor::with_shape(x.get_value().to_vec(), x.get_shape().to_vec()))
            },
            _ => Err(TensorError::Unimplemented("Type Error".to_owned()))
        }
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
    fn get_tensor(res: TensorOperationResult, wanted_shape: &Shape) -> Result<Tensor<Self::BindingType>, TensorError> {
        match res {
            TensorOperationResult::UInt(x) => {
                if !x.matches_shape(wanted_shape) {return Err(TensorError::ShapeError("Result Shapes didn't match".to_owned()))}
                Ok(Tensor::with_shape(x.get_value().to_vec(), x.get_shape().to_vec()))
            },
            _ => Err(TensorError::Unimplemented("Type Error".to_owned()))
        }
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
    fn get_tensor(res: TensorOperationResult, wanted_shape: &Shape) -> Result<Tensor<Self::BindingType>, TensorError> {
        match res {
            TensorOperationResult::Float(x) => {
                if !x.matches_shape(wanted_shape) {return Err(TensorError::ShapeError("Result Shapes didn't match".to_owned()))}
                Ok(Tensor::with_shape(x.get_value().to_vec(), x.get_shape().to_vec()))
            },
            _ => Err(TensorError::Unimplemented("Type Error".to_owned()))
        }
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
    fn get_tensor(res: TensorOperationResult, wanted_shape: &Shape) -> Result<Tensor<Self::BindingType>, TensorError> {
        match res {
            TensorOperationResult::Double(x) => {
                if !x.matches_shape(wanted_shape) {return Err(TensorError::ShapeError("Result Shapes didn't match".to_owned()))}
                Ok(Tensor::with_shape(x.get_value().to_vec(), x.get_shape().to_vec()))
            },
            _ => Err(TensorError::Unimplemented("Type Error".to_owned()))
        }
    }
}