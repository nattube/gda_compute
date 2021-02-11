use std::cell::RefCell;
use std::rc::Rc;

use super::{SupportedDataTypes, Tensor, TensorBinding, INPUT_NAME};

impl<'a> TensorBinding<'a> {
    pub(crate) fn from_tensor<T>(tensor: &'a Tensor<T>, id: u32) -> Self
    where T: SupportedDataTypes + SupportedDataTypes<BindingType = T>, {
        TensorBinding {id, value: T::to_data_holder(tensor), change: Rc::new(RefCell::new(tensor.get_change()))}
    }

    pub(crate) fn has_changed(&self) -> bool {
        if self.value.get_change() > *self.change.borrow() {
            *self.change.borrow_mut() = self.value.get_change();
            return true
        }
        false
    }

    pub(crate) fn get_type_glsl(&self) -> String {
        self.value.get_type_glsl()
    }

    pub(crate) fn get_value_glsl(&self, index: &str) -> String {
        if self.value.is_single() {format!("{}{}", INPUT_NAME, self.id)} else {format!("{}{}[{}]", INPUT_NAME, self.id, index)}
    }

    pub(crate) fn copy(&self) -> Self {
        TensorBinding {id: self.id, value: self.value.copy(), change: Rc::new(RefCell::new(*self.change.borrow()))}
    }
}