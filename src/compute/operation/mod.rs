use std::cmp;
use std::collections::HashMap;
use super::instruction::*;


pub enum Identifier {
    Value(usize),
    SingleResult(usize, Identifier, Box<dyn SingleInstruction>),
    DualResult(usize, Identifier, Identifier, Box<dyn DualInstruction>)
}

pub struct LocalScope {
    global: HashMap<String, Identifier>,
    local: HashMap<String, Identifier>,
}

pub struct FunctionBuilder {
    scope: LocalScope,
    returns: String,
    arguments: Vec<Identifier>,

}

pub struct OperationBuilder {
    
}

impl OperationBuilder {

}


pub trait TOperation {
    pub fn takes() -> usize;
    pub fn is_valid(shapes: Vec<Shape>) -> bool;
    pub fn result_shape(shapes: Vec<Shape>) -> Shape;
    pub fn indexing_function(shapes: Vec<Shape>, scope: LocalScope) -> Option<Vec<Instructions>>;
    pub fn max_steps(shapes: Vec<Shape>) -> (usize, usize, usize);
    pub fn build(inputs: Vec<Instructions>) -> Instructions;
}

pub struct MatAdd {}

impl TOperation for MatAdd {
    pub fn takes() -> usize { 2 }
    pub fn is_valid(shapes: Vec<Shape>) -> bool {
           shapes[0].len() >= shapes[1].len() 
        && !shapes[1].iter().zip(shapes[0]).any(|(x,y)| x != y)
    }
    pub fn result_shape(shapes: Vec<Shape>) -> Shape {
        shape[0]
    }
    pub fn indexing_function(shapes: Vec<Shape>, ctx: &mut LocalScope) -> Option<Vec<Instructions>>{
        None
    }
    pub fn max_steps(shapes: Vec<Shape>) -> (usize, usize, usize) {
        (shapes[0].iter().product(), 1, 1)
    }
    pub fn build(inputs: Vec<Instructions>, ctx: &mut LocalScope) -> Instructions {
        IAdd.build(inputs)
    }
}

pub struct MatMul {}

impl TOperation for MatMul {
    pub fn is_valid(shape1: Shape, shape2: Shape) -> bool {
        if shape1.len() < 2 || shape2.len() < 1 || shape1[1] != shape2[0] {
            return false
        }

        let mut i = 2;


        loop {
            if i >= shape1.len() || i >= shape2.len() {
                break;
            }
            if shape1[i] != shape2[i] {
                return false;
            }
            i++;
        }

        true
    }

    pub fn result_shape(shape1: Shape, shape2: Shape) -> Shape {
        if shape2.len() < 2 {
            return vec![shape1[0]]
        }
        
        let mut result = vec![shape1[0], shape2[1]];

        if cmp::max(shape1.len(), shape2.len()) <= 2 {
            return result
        }

        if shape1.len() > shape2.len() {
            for i in 2..shape1.len() {
                result.push(shape1[i])
            }
        }

        else {
            for i in 2..shape2.len() {
                result.push(shape2[i])
            }
        }

        return result
    }

    pub fn array_order(shape1: Shape, shape2: Shape) -> (Vec<usize>, Vec<usize>) {
        
    }

    pub fn build(ctx: LocalScope) {
        let t2 = ctx.get("tensor1");
        let t1 = ctx.get("tensor2");
        let mut result = ctx.get("result");

        let cnst = ctx.define_const("by_two", 2);
        let inter = ctx.set("intermediat", t1[step] + t2[step]);

        result[step] = inter * cnst;
    }
}