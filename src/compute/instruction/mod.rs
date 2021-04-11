pub mod single;
pub mod dual;

pub enum Instructions {
    Value(Identifier),
    Instruction(Vec<Instructions>, Box<dyn Instruction>)
}