use std::f64::EPSILON;
use crate::utils::{print_matrix, reshape2D};
pub struct MyLinearRegression_rb {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
}

impl MyLinearRegression_rb {
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        MyLinearRegression_rb {
            weights: vec![vec![0.0; n_features]; n_classes],
            biases: vec![0.0; n_classes],
        }
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, learning_rate: f64, epochs: usize) {
        let n_features = X_train[0].len();
        let n_classes = y_train[0].len();

        for epoch in 0..epochs {
            let mut total_error = 0.0;
            for (x, y) in X_train.iter().zip(y_train.iter()) {
                let predictions = self.predict_single(x);
                let errors: Vec<f64> = y.iter().zip(predictions.iter()).map(|(&y_i, &pred_i)| y_i - pred_i).collect();

                // Calculate the squared error for the current prediction
                let squared_error: f64 = errors.iter().map(|&e| e * e).sum();
                total_error += squared_error;

                for (c, error) in errors.iter().enumerate() {
                    for j in 0..n_features {
                        self.weights[c][j] += learning_rate * error * x[j];
                    }
                    self.biases[c] += learning_rate * error;
                }
            }

            // Average MSE over the entire training dataset
            let mse = total_error / (X_train.len() as f64 * n_classes as f64);
            
            // println!("Epoch {}: MSE = {}", epoch + 1, mse);
        }
    }

    fn predict_single(&self, x: &Vec<f64>) -> Vec<f64> {
        let mut linear_outputs: Vec<f64> = self.biases.clone();
        for (class, class_weights) in self.weights.iter().enumerate() {
            for (weight, &feature) in class_weights.iter().zip(x.iter()) {
                linear_outputs[class] += weight * feature;
            }
        }
        self.softmax(&linear_outputs)
    }

    fn softmax(&self, logits: &Vec<f64>) -> Vec<f64> {
        let max_logit = logits.iter().cloned().fold(f64::NEG_INFINITY, f64::max);
        let exps: Vec<f64> = logits.iter().map(|&logit| (logit - max_logit).exp()).collect();
        let sum_exps: f64 = exps.iter().sum();
        exps.iter().map(|&exp| exp / (sum_exps + EPSILON)).collect()
    }

    pub fn predict(&self, x_test: Vec<Vec<f64>>) -> Vec<Vec<f64>> {
        x_test.iter().map(|x| self.predict_single(x)).collect()
    }
}

#[no_mangle]
pub extern "C" fn create_MyLinearRegression_rb(n_features: i32, n_classes: i32) -> *mut MyLinearRegression_rb {
    let model = MyLinearRegression_rb::new(n_features as usize, n_classes as usize);
    let boxed_model = Box::new(model);
    Box::into_raw(boxed_model)
}

#[no_mangle]
pub extern "C" fn train_MyLinearRegression_rb(p_model: *mut MyLinearRegression_rb,

                                            p_x_train: *const f64,
                                            x_train_shape_0: i32, x_train_shape_1: i32,

                                            p_y_train: *const f64,
                                            y_train_shape_0: i32, y_train_shape_1: i32,

                                            learning_rate: f64,
                                            epochs: i32) {
    let model = unsafe { &mut *p_model };

    let flatten_x_train = unsafe {
        std::slice::from_raw_parts(p_x_train, (x_train_shape_0 * x_train_shape_1) as usize)
    }.to_vec();
    let x_train = reshape2D(flatten_x_train, (x_train_shape_0 as usize, x_train_shape_1 as usize));

    let flatten_y_train = unsafe {
        std::slice::from_raw_parts(p_y_train, (y_train_shape_0 * y_train_shape_1) as usize)
    }.to_vec();
    let y_train = reshape2D(flatten_y_train, (y_train_shape_0 as usize, y_train_shape_1 as usize));
    model.train(x_train, y_train, learning_rate, epochs as usize);
}

#[no_mangle]
pub extern "C" fn predict_MyLinearRegression_rb(p_model: *mut MyLinearRegression_rb,

                                                p_input: *const f64,
                                                input_shape_0: i32, input_shape_1: i32,
) -> *const f64 {
    let model = unsafe { &mut *p_model };

    let flatten_input = unsafe {
        std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = model.predict(input);

    let mut flatten_pred = vec![];
    for row in predictions.iter() {
        for &value in row.iter() {
            flatten_pred.push(value);
        }
    }

    let leaked_predictions = Vec::leak(flatten_pred);
    leaked_predictions.as_ptr()
}