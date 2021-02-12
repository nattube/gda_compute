pub mod processor;
pub mod tensor;

use tensor::{Tensor, TensorError};
use processor::{Processor, ProcessorSelectionConstraint};

pub fn run() -> Vec<u32> {
    //build_shader();
    //gpu::public_run()
    Vec::new()
}

pub fn build_shader() {
/*    let a = Tensor::new(vec![1f32,1.0,1.0]);
    let b = Tensor::new(vec![1f32,1.0,1.0]);
    let op = &a + &b;
    let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
    let shader = gpu.build(op);
    let res1 = gpu.execute(&shader);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res1);

    if let Err(TensorError::ShapeError(x)) = b.change_value(vec![4f32,3.0,2.0]) {
        println!("{}", x);
    }
    let res2 = gpu.execute(&shader);
    println!("{:?} + {:?} = {:?}", a.get_value(), b.get_value(), res2); */
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    pub fn test_add_1() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1f32, 2.0, 3.0]);
        let b = Tensor::new(vec![1f32]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a + &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], 2f32);
        assert_eq!(v_result[1], 3f32);
        assert_eq!(v_result[2], 4f32);

        Ok(())
    }

    #[test]
    pub fn test_add_2() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1f32, 2.0, 3.0]);
        let b = Tensor::new(vec![2f32]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a + &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], 3f32);
        assert_eq!(v_result[1], 4f32);
        assert_eq!(v_result[2], 5f32);

        Ok(())
    }

    #[test]
    pub fn test_add_t() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1f32, 2.0, 3.0]);
        let b = Tensor::new(vec![1f32, 2.0, 3.0]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a + &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], 2f32);
        assert_eq!(v_result[1], 4f32);
        assert_eq!(v_result[2], 6f32);

        Ok(())
    }


    #[test]
    pub fn test_subt_1() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1f32, 2.0, 3.0]);
        let b = Tensor::new(vec![1f32]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a - &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], 0f32);
        assert_eq!(v_result[1], 1f32);
        assert_eq!(v_result[2], 2f32);

        Ok(())
    }

    #[test]
    pub fn test_subt_2() -> Result<(), TensorError> {
        let a = Tensor::new(vec![1f32, 2.0, 3.0]);
        let b = Tensor::new(vec![2f32]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a - &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], -1f32);
        assert_eq!(v_result[1], 0f32);
        assert_eq!(v_result[2], 1f32);

        Ok(())
    }

    #[test]
    pub fn test_subt_t() -> Result<(), TensorError> {
        let a = Tensor::new(vec![2f32, 4.0, 17.0]);
        let b = Tensor::new(vec![1f32, 2.0, 3.0]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let op = &a - &b;        

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], 1f32);
        assert_eq!(v_result[1], 2f32);
        assert_eq!(v_result[2], 14f32);

        Ok(())
    }


    #[test]
    pub fn test_complex_op() -> Result<(), TensorError> {
        let a = Tensor::new(vec![2f32, 4.0, 17.0]);
        let b = Tensor::new(vec![1f32, 2.0, 3.0]);
        let c = Tensor::new(vec![1f32, 1.0, 4.0]);
        let d = Tensor::new(vec![1f32, 2.0, 3.0]);
        let e = Tensor::new(vec![1f32]);

        let result_tensor: Tensor<f32> = Tensor::zeros_from_shape(vec![3]);
    
        let mut op = &a + &b - &c;      
        let op2 = &d - &e;
        op = op - op2;  

        let mut gpu = Processor::new(ProcessorSelectionConstraint::None);
        
        let shader = gpu.build(op, result_tensor);
        let res1 = gpu.execute(&shader);

        let v_result = res1?.get_value().to_vec();

        assert_eq!(v_result[0], (2f32 + 1f32 - 1f32) - (1f32 - 1f32));
        assert_eq!(v_result[1], (4f32 + 2f32 - 1f32) - (2f32 - 1f32));
        assert_eq!(v_result[2], (17f32 + 3f32 - 4f32) - (3f32 - 1f32));

        Ok(())
    }


    #[test]
    pub fn test_tensor_indexing_1() {
        let a = Tensor::new(vec![0f32, 1.0, 2.0, 3.0]);

        assert_eq!(a.get(&[0]), 0f32);
        assert_eq!(a.get(&[1]), 1f32);
        assert_eq!(a.get(&[2]), 2f32);
        assert_eq!(a.get(&[3]), 3f32);
    }

    #[test]
    pub fn test_tensor_indexing_2() {
        let a = Tensor::with_shape(vec![0f32, 0.1, 0.2, 1.0, 1.1, 1.2], vec![2, 3]);

        assert_eq!(a.get(&[0,0]), 0f32);
        assert_eq!(a.get(&[0,1]), 0.1f32);
        assert_eq!(a.get(&[0,2]), 0.2f32);
        assert_eq!(a.get(&[1,0]), 1.0f32);
        assert_eq!(a.get(&[1,1]), 1.1f32);
        assert_eq!(a.get(&[1,2]), 1.2f32);
    }
}