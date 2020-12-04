pub mod data;

use data::datapoint::*;

static mut DATA_CACHE: Vec<DataPoint> = Vec::new();

pub fn load_data(data: String) {
    let split = data.split(";");
    let mut datapoints: Vec<DataPoint> = Vec::new();
    for s in split {
        datapoints.push(match_data(s));
    }

    unsafe {
        DATA_CACHE.append(&mut datapoints);  
    }
}

pub fn get_data() -> Vec<DataPoint> {
    let result: Vec<DataPoint>;
    unsafe {
        result = DATA_CACHE.to_vec();
    }

    result
}