use serde::{Serialize, Deserialize};

#[derive(Clone, Serialize, Deserialize)]
#[serde(untagged)]
pub enum DataPoint {
    Bool(bool),
    Float(f64),
    Int(i64),
    Str(String)
}

pub trait ToStringVec {
    fn into_str_vec(&self) -> Vec<String>;
}

impl ToStringVec for Vec<DataPoint> {
    fn into_str_vec(&self) -> Vec<String> {
        let mut out :Vec<String> = Vec::with_capacity(self.len());

        for d in self {
            match d {
                DataPoint::Bool(x) => out.push((*x).to_string()),
                DataPoint::Float(x) => out.push((*x).to_string()),
                DataPoint::Int(x) => out.push((*x).to_string()),
                DataPoint::Str(x) => out.push((*x).to_owned())
            }
        }

        out
    }
}

pub fn match_data(data: &str) -> DataPoint {
    let is_bool = data.parse::<bool>();
    let is_float = data.parse::<f64>();
    let is_int = data.parse::<i64>();

    if let Ok(x) = is_bool {
        return DataPoint::Bool(x)
    }
    else if let Ok(x) = is_float {
        return DataPoint::Float(x)
    }
    else if let Ok(x) = is_int {
        return DataPoint::Int(x)
    }

    DataPoint::Str(String::from(data))
}