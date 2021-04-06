use std::cmp;

pub trait TOperation {
    pub fn is_valid(shape1: Shape, shape2: Shape) -> bool;
    pub fn result_shape(shape1: Shape, shape2: Shape) -> Shape;
    pub fn array_order(shape1: Shape, shape2: Shape) -> (Vec<usize>, Vec<usize>);
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
}