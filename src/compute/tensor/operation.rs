use std::fmt::Write;
use std::cmp;

use std::ops::{Add};

use super::{Operation, SupportedDataTypes, TensorOperationResult, Tensor, TensorBinding, TwoValueOperation, INPUT_NAME};

impl<'a> Operation<'a> {
    pub(crate) fn get_last_binding(&self) -> u32 {
        match self {
            Operation::Var(x) => x.id,
            Operation::DualOp {left, right, ..} => cmp::max(left.get_last_binding(), right.get_last_binding()),
            _ => unimplemented!()
        }
    }

    pub(crate) fn reset_binding_from(&mut self, start: u32) -> u32 {
        match self {
            Operation::Var(x) => { x.id = start; return start+1 },
            Operation::DualOp {left, right, ..} => {
                let newStart = left.reset_binding_from(start);
                return right.reset_binding_from(newStart)
            },
            _ => unimplemented!()
        }
    }

    pub(crate) fn contains_tensor<T>(&self, tensor: &Tensor<T>) -> Option<TensorBinding<'a>>
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
        match self {
            Operation::Var(x) => {
                //x. as *const _ == ref2 as *const _
                if T::to_data_holder(tensor) == x.value {return Some(x.copy())}
                None
            }
            Operation::DualOp {left, right, ..} => {
                if let Some(x) = left.contains_tensor(tensor) {return Some(x)}
                if let Some(x) = right.contains_tensor(tensor) {return Some(x)}
                None
            }
            _ => None
        }
    }
}

impl<'a> Operation<'a> {
    pub(crate) fn build_gpu(&self) -> (Vec<u8>, Vec<TensorBinding<'a>>, u32) {        
        let mut inputs = Vec::<TensorBinding<'a>>::new();
        self.fill_input(&mut inputs);
        
        let mut s = String::new();
        writeln!(&mut s, "#version 450");
        writeln!(&mut s, "layout(local_size_x = 1) in;");
        
        writeln!(&mut s);
        
        let mut binding: u32 = 0;

        for i in &inputs {
            writeln!(&mut s, "readonly layout(set = 0, binding = {}) buffer b{} {{", (*i).id, (*i).id);
            writeln!(&mut s, "{} {}{};", (*i).get_type_glsl(), INPUT_NAME, (*i).id);
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
        writeln!(&mut s, "result[index] = {};", self.build_equation());
        writeln!(&mut s, "}}");

        println!("{}", s);

        (Operation::build_shader(&s), inputs, binding)
    }

    fn build_equation(&self) -> String {
        match self {
            Operation::Var(x) => {
                x.get_value_glsl("index")
            }
            Operation::DualOp {left, right, result, op} => {
                format!("({} {:?} {})", left.build_equation(), op, right.build_equation())
            }
            _ => "".to_owned()
        }        
    }

    fn fill_input(&self, inputs: &mut Vec<TensorBinding<'a>>) {
        match self {
            Operation::Var(x) => {
                if !inputs.iter().any(|y| y.id == x.id) {
                    inputs.push(x.copy());
                }
            },
            Operation::SingleOp {value, result, op}  => {
                value.fill_input(inputs);
            },
            Operation::DualOp {left, right, result, op} => {
                left.fill_input(inputs);
                right.fill_input(inputs);
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

impl<'a, T> Add<&'a Tensor<T>> for Operation<'a>
where T: SupportedDataTypes + SupportedDataTypes<BindingType = T> {
    type Output = Operation<'a>;

    fn add(self, tensor: &'a Tensor<T>) -> Operation<'a> {
        let result = TensorOperationResult::from_1_and_op(tensor, &self, TwoValueOperation::Add);
        let binding = self.get_last_binding() + 1;

        let tensor = match self.contains_tensor(tensor) {
            Some(x) => x.copy(),
            None => TensorBinding::from_tensor(tensor, binding)
        };

        Operation::DualOp {
            left: Box::new(Operation::Var(Box::new(tensor))), 
            right: Box::new(self),
            result,
            op: TwoValueOperation::Add
        }
    }
}

impl<'a> Add<Operation<'a>> for Operation<'a> {
    type Output = Operation<'a>;

    fn add(self, mut op: Operation<'a>) -> Operation<'a> {
        let result = TensorOperationResult::from_2_ops(&op, &self, TwoValueOperation::Add);
        let binding = self.get_last_binding();
        op.reset_binding_from(binding+1);

        Operation::DualOp {
            left: Box::new(op), 
            right: Box::new(self),
            result,
            op: TwoValueOperation::Add
        }
    }
}