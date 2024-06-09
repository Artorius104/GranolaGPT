extern crate nalgebra as na;
use crate::utils::{reshape2D};

use na::{DMatrix};

pub struct MyLinearRegression {
    weights: Vec<Vec<f64>>,
}

impl MyLinearRegression {
    pub fn new() -> Self {
        MyLinearRegression { 
            weights: vec![] 
        }
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, y: Vec<Vec<f64>>) {
        let X_train_matrix = DMatrix::from_vec(X_train.len(), X_train[0].len(), X_train.iter().flatten().cloned().collect());
        let y_matrix = DMatrix::from_vec(y.len(), y[0].len(), y.iter().flatten().cloned().collect());


        let x_pseudo_inverse = (X_train_matrix.clone().transpose() * X_train_matrix.clone()).try_inverse().unwrap() * X_train_matrix.clone().transpose();
        let weights_matrix = x_pseudo_inverse * y_matrix;

        self.weights = weights_matrix.row_iter().map(|row| row.iter().cloned().collect()).collect();
    }

    pub fn predict(&self, input: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        let input_matrix = DMatrix::from_vec(input.len(), input[0].len(), input.iter().flatten().cloned().collect());
        let weights_matrix = DMatrix::from_vec(self.weights.len(), self.weights[0].len(), self.weights.iter().flatten().cloned().collect());

        let predictions = input_matrix * weights_matrix;

        let mut predictions:Vec<Vec<f64>> = predictions.row_iter().map(|row| row.iter().cloned().collect()).collect();
        predictions
    }
}

#[no_mangle]
pub extern "C" fn create_MyLinearRegression() -> *mut MyLinearRegression{
    let model = MyLinearRegression::new();
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model 
}

#[no_mangle]
pub extern "C" fn train_MyLinearRegression(p_model:*mut MyLinearRegression,

                                p_X_train:*const f64, 
                                X_train_shape_0:i32, X_train_shape_1:i32,

                                p_y_train:*const f64, 
                                y_train_shape_0:i32, y_train_shape_1:i32){

    let mut model = unsafe {&mut *p_model};

    let flatten_X_train = unsafe {
        {std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)}
    }.to_vec();
    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let flatten_y_train = unsafe {
        {std::slice::from_raw_parts(p_y_train, (y_train_shape_0 * y_train_shape_1) as usize)}
    }.to_vec();
    let y_train = reshape2D(flatten_y_train, (y_train_shape_0 as usize, y_train_shape_1 as usize));
    
    model.train(X_train, y_train);                                   
}

#[no_mangle]
pub extern "C" fn predict_MyLinearRegression(p_model:*mut MyLinearRegression,
                                    
                                p_input:*const f64, 
                                input_shape_0:i32, input_shape_1:i32) -> *const f64{

    let mut model = unsafe {&mut *p_model};

    let flatten_input = unsafe {
        {std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)}
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = model.predict(input);

    let mut flatten_pred = vec![];
    for i in 0..predictions.len(){
        for j in 0..predictions[0].len(){
            flatten_pred.push(predictions[i][j].clone());
        }
    }

    let leaked_predictions = Vec::leak(flatten_pred);
    leaked_predictions.as_ptr()
}