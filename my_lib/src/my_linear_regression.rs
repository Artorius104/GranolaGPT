extern crate nalgebra as na;
use na::{DMatrix, DVector};

use crate::utils::chunk_vector;
use crate::utils::matrix_from_2d_vec;

pub struct MyLinearRegression {
    weights: DMatrix<f64>,
}

impl MyLinearRegression {
    pub fn new(input_size: usize, output_sizes: usize) -> Self {
        MyLinearRegression {
            weights: DMatrix::zeros(input_size, output_sizes),
        }
    }

    pub fn train(&mut self, X: &DMatrix<f64>, y: &DMatrix<f64>) {
        let X_transpose = X.transpose();
        let pseudo_inverse = (X_transpose.clone() * X.clone()).try_inverse().unwrap() * X_transpose;
        self.weights = pseudo_inverse * y;
    }

    pub fn predict(&self, X: &DMatrix<f64>) -> DMatrix<f64> {
        X * &self.weights
    }
}

#[no_mangle]
pub extern "C" fn create_MyLinearRegression(input_size: i32, output_size: i32) -> *mut MyLinearRegression {
    let model = MyLinearRegression::new(input_size as usize, output_size as usize);

    let boxed_model = Box::new(model);
    let leaked_boxed_model = Box::leak(boxed_model);
    leaked_boxed_model
}

#[no_mangle]
pub extern "C" fn train_MyLinearRegression(
                            p_model: *mut MyLinearRegression,

                            p_X:*const f64,
                            input_size: i32,

                            p_y:*const f64,
                            output_size : i32,

                            n_samples: i32){

    let model = unsafe {&mut *p_model};

    let X_flatten = unsafe {
        {std::slice::from_raw_parts(p_X, (input_size * n_samples) as usize)}
    };
    let X = chunk_vector(X_flatten.to_vec(), input_size as usize);

    let y_flatten = unsafe {
        {std::slice::from_raw_parts(p_y, (output_size * n_samples) as usize)}
    };
    let y = chunk_vector(y_flatten.to_vec(), output_size as usize);

    let X_matrix = matrix_from_2d_vec(&X);
    let y_matrix = matrix_from_2d_vec(&y);
}

#[no_mangle]
pub extern "C" fn predict_MyLinearRegresion(p_model:*mut MyLinearRegression, 
                                p_samples:*const f64, 
                                input_size:i32, 
                                n_samples:i32) -> *const f64{

    let mut predictions:Vec<f64> = vec![];
    let model = unsafe{&mut *p_model};

    let samples_flatten = unsafe {
        std::slice::from_raw_parts(p_samples, (input_size * n_samples) as usize)
    };

    let samples = chunk_vector(samples_flatten.to_vec(), input_size as usize);
    
    let samples_matrix = matrix_from_2d_vec(&samples);

    let predictions_matrix = model.predict(&samples_matrix);

    let mut predictions = Vec::new();
    for i in 0..predictions_matrix.nrows() {
        for j in 0..predictions_matrix.ncols() {
            predictions.push(predictions_matrix[(i, j)]);
        }
    }

    let leaked_predictions = Vec::leak(predictions.to_vec());
    leaked_predictions.as_ptr()     
}