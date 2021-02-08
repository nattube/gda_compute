use std::fmt::Write;
use std::cell::RefCell;
use std::cell::Ref;
use std::rc::Rc;
use std::ops::{Add};

/* 

pub enum TensorBindingHolder<'a> {
    Int(Box<TensorBinding<'a, i32>>),
    Uint(Box<TensorBinding<'a, u32>>),
    Float(Box<TensorBinding<'a, f32>>),
    Double(Box<TensorBinding<'a, f64>>),
    Boolean(Box<TensorBinding<'a, bool>>)
}

impl<'a> From<TensorBinding<'a, i32>> for TensorBindingHolder<'a> {
    fn from(val: TensorBinding<i32>) -> Self {
        TensorBindingHolder::Int(Box::new(val))
    }
}

impl<'a> From<TensorBinding<'a, u32>> for TensorBindingHolder<'a> {
    fn from(val: TensorBinding<u32>) -> Self {
        TensorBindingHolder::Uint(Box::new(val))
    }
}

impl<'a> From<TensorBinding<'a, f32>> for TensorBindingHolder<'a> {
    fn from(val: TensorBinding<f32>) -> Self {
        TensorBindingHolder::Float(Box::new(val))
    }
}

impl<'a> From<TensorBinding<'a, f64>> for TensorBindingHolder<'a> {
    fn from(val: TensorBinding<f64>) -> Self {
        TensorBindingHolder::Double(Box::new(val))
    }
}

impl<'a> From<TensorBinding<'a, bool>> for TensorBindingHolder<'a> {
    fn from(val: TensorBinding<bool>) -> Self {
        TensorBindingHolder::Boolean(Box::new(val))
    }
} 

*/

pub trait SupportedDataTypes {}
impl SupportedDataTypes for i32 {}
impl SupportedDataTypes for u32 {}
impl SupportedDataTypes for f32 {}
impl SupportedDataTypes for f64 {}
impl SupportedDataTypes for bool {}

pub struct TensorBinding<'a, T> {
    pub(crate) id: u32,
    pub(crate) value: &'a Tensor<T>,
    pub(crate) change: Rc<RefCell<u32>>
}

impl<'a, T> TensorBinding<'a, T> 
where T: SupportedDataTypes {
    pub(crate) fn has_changed(&self) -> bool {
        if self.value.get_change() > *self.change.borrow() {
            *self.change.borrow_mut() = self.value.get_change();
            return true
        }
        false
    }
}

pub struct Tensor<T> {
    pub(crate) value: RefCell<Vec<T>>,
    pub(crate) change: Rc<RefCell<u32>>
    //shape: Vec<u32>
}

impl<T> Tensor<T>
where T: SupportedDataTypes {
    pub fn new(vec: Vec<T>) -> Tensor<T> {
        Tensor {value: RefCell::new(vec), change: Rc::new(RefCell::new(0))}
    }

    pub fn get_value(&self) -> Ref<Vec<T>> {
        self.value.borrow()
    }

    pub fn set_value(&self, mut val: Vec<T>) {
        let mut old_val = self.value.borrow_mut();
        old_val.clear();
        old_val.append(&mut val);
        *self.change.borrow_mut() += 1;
    }

    pub(crate) fn get_change(&self) -> u32 {
        *self.change.borrow()
    }
}

impl<'a, T, U> Add<&Tensor<T>> for &'a Tensor<U> 
where T: SupportedDataTypes, 
      U: SupportedDataTypes {
    type Output = Operation<'a>;

    fn add(self, other: &Tensor<T>) -> Operation<'a> {
        Operation::Add(
            Box::new(Operation::Var(Box::new(TensorBinding {id: 0, value: self, change: Rc::new(RefCell::new(self.get_change()))}))), 
            Box::new(Operation::Var(Box::new(TensorBinding {id: 1, value: other, change: Rc::new(RefCell::new(other.get_change()))})))
        )
    }
}

pub struct Ubs {

}

pub enum Operation<'a> {    
    Int(Box<TensorBinding<'a, i32>>),
    Add(Box<Operation<'a>>, Box<Operation<'a>>),
    Mult(Box<Operation<'a>>, Box<Operation<'a>>)
}

impl<'a> Operation<'a> {
    pub(crate) fn build_gpu(&'a self) -> (Vec<u8>, Vec<&'a TensorBinding<'a>>, u32) {        
        let mut inputs = Vec::<&'a TensorBinding<'a>>::new();
        self.fill_input(&mut inputs);
        
        let mut s = String::new();
        writeln!(&mut s, "#version 450");
        writeln!(&mut s, "layout(local_size_x = 1) in;");
        
        writeln!(&mut s);
        
        let mut binding: u32 = 0;

        for i in &inputs {
            writeln!(&mut s, "readonly layout(set = 0, binding = {}) buffer b{} {{", (*i).id, (*i).id);
            writeln!(&mut s, "float[] indices{};", (*i).id);
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
        writeln!(&mut s, "{}", self.build_str());
        writeln!(&mut s, "}}");

        println!("{}", s);

        (Operation::build_shader(&s), inputs, binding)
    }

    fn build_str(&self) -> &str {
        "result[index] = indices0[index] * indices1[index];"
    }

    fn fill_input(&'a self, inputs: &mut Vec<&'a TensorBinding<'a>>) {
        match self {
            Operation::Var(x) => {
                if !inputs.iter().any(|&y| y.id == x.id) {
                    inputs.push(&x);
                }
            },
            Operation::Add(x, y) | Operation::Mult(x, y) => {
                x.fill_input(inputs);
                y.fill_input(inputs);
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