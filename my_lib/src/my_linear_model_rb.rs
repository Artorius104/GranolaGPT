use std::f64::EPSILON;
use crate::utils::{print_matrix, reshape2D};

pub struct MyLinearModel_rb {
    weights: Vec<Vec<f64>>,
    biases: Vec<f64>,
    train_losses: Vec<f64>,
    test_losses: Vec<f64>,
}

impl MyLinearModel_rb {
    pub fn new(n_features: usize, n_classes: usize) -> Self {
        MyLinearModel_rb {
            weights: vec![vec![0.0; n_features]; n_classes],
            biases: vec![0.0; n_classes],
            train_losses: vec![],
            test_losses: vec![],
        }
    }

    pub fn train(&mut self, 
        X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, 
        X_test: Vec<Vec<f64>>, y_test: Vec<Vec<f64>>, 
        learning_rate: f64, epochs: usize, 
        is_classification: bool) {
        let n_features = X_train[0].len();
        let n_classes = y_train[0].len();

        for epoch in 0..epochs {
        let mut total_error = 0.0;  
        for i in 0..X_train.len() {
            let pred = self.predict_single(&X_train[i], is_classification);
            let mut error = vec![0.0; n_classes];
            for j in 0..n_classes {
                error[j] = pred[j] - y_train[i][j];
            }

            for j in 0..n_classes {
                for k in 0..n_features {
                    self.weights[j][k] -= learning_rate * error[j] * X_train[i][k];
                }
                self.biases[j] -= learning_rate * error[j];
            }
            total_error += mean_squared_error(&y_train[i], &pred);
        }
        self.train_losses.push(total_error / X_train.len() as f64);

        // Calcul de l'erreur sur les données de test
        let mut test_error = 0.0;
        for i in 0..X_test.len() {
            let pred = self.predict_single(&X_test[i], is_classification);
            test_error += mean_squared_error(&y_test[i], &pred);
        }
        self.test_losses.push(test_error / X_test.len() as f64);

        println!("Epoch {}: Train Loss: {}, Test Loss: {}", epoch, self.train_losses.last().unwrap(), self.test_losses.last().unwrap());
        }
    }

    fn predict_single(&self, x: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        let mut linear_outputs: Vec<f64> = self.biases.clone();
        for (class, class_weights) in self.weights.iter().enumerate() {
            for (weight, &feature) in class_weights.iter().zip(x.iter()) {
                linear_outputs[class] += weight * feature;
            }
        }
        if is_classification {
            linear_outputs.iter().map(|&x| x.tanh()).collect()
        } else {
            linear_outputs
        }
    }

    pub fn predict(&self, X_test: Vec<Vec<f64>>, is_classification: bool) -> Vec<Vec<f64>> {
        X_test.iter().map(|x| self.predict_single(x, is_classification)).collect()
    }
}

fn mean_squared_error(y_true: &Vec<f64>, y_pred: &Vec<f64>) -> f64 {
    y_true.iter().zip(y_pred.iter()).map(|(t, p)| (t - p).powi(2)).sum::<f64>() / y_true.len() as f64
}

#[no_mangle]
pub extern "C" fn create_MyLinearModel_rb(n_features: i32, n_classes: i32) -> *mut MyLinearModel_rb {
    let model = MyLinearModel_rb::new(n_features as usize, n_classes as usize);
    let boxed_model = Box::new(model);
    Box::into_raw(boxed_model)
}

#[no_mangle]
pub extern "C" fn train_MyLinearModel_rb(p_model: *mut MyLinearModel_rb,

                                         p_X_train: *const f64,
                                         X_train_shape_0: i32, X_train_shape_1: i32,

                                         p_y_train: *const f64,
                                         y_train_shape_0: i32, y_train_shape_1: i32,

                                         p_X_test: *const f64,
                                         X_test_shape_0: i32, X_test_shape_1: i32,

                                         p_y_test: *const f64,
                                         y_test_shape_0: i32, y_test_shape_1: i32,

                                         learning_rate: f64,
                                         epochs: i32,
                                         is_classification: bool) {
    let model = unsafe { &mut *p_model };

    let flatten_X_train = unsafe {
        std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)
    }.to_vec();
    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let flatten_y_train = unsafe {
        std::slice::from_raw_parts(p_y_train, (y_train_shape_0 * y_train_shape_1) as usize)
    }.to_vec();
    let y_train = reshape2D(flatten_y_train, (y_train_shape_0 as usize, y_train_shape_1 as usize));

    let flatten_X_test = unsafe {
        std::slice::from_raw_parts(p_X_test, (X_test_shape_0 * X_test_shape_1) as usize)
    }.to_vec();
    let X_test = reshape2D(flatten_X_test, (X_test_shape_0 as usize, X_test_shape_1 as usize));

    let flatten_y_test = unsafe {
        std::slice::from_raw_parts(p_y_test, (y_test_shape_0 * y_test_shape_1) as usize)
    }.to_vec();
    let y_test = reshape2D(flatten_y_test, (y_test_shape_0 as usize, y_test_shape_1 as usize));

    model.train(X_train, y_train, X_test, y_test, learning_rate, epochs as usize, is_classification);
}

#[no_mangle]
pub extern "C" fn predict_MyLinearModel_rb(p_model: *mut MyLinearModel_rb,

                                           p_input: *const f64,
                                           input_shape_0: i32, input_shape_1: i32,
                                           is_classification: bool) -> *const f64 {
    let model = unsafe { &mut *p_model };

    let flatten_input = unsafe {
        std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = model.predict(input, is_classification);

    let mut flatten_pred = vec![];
    for row in predictions.iter() {
        for &value in row.iter() {
            flatten_pred.push(value);
        }
    }

    let leaked_predictions = Vec::leak(flatten_pred);
    leaked_predictions.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_train_losses_MyLinearModel_rb(p_model: *mut MyLinearModel_rb) -> *const f64 {
    let model = unsafe { &mut *p_model };
    let leaked_losses = Vec::leak(model.train_losses.clone());
    leaked_losses.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_test_losses_MyLinearModel_rb(p_model: *mut MyLinearModel_rb) -> *const f64 {
    let model = unsafe { &mut *p_model };
    let leaked_losses = Vec::leak(model.test_losses.clone());
    leaked_losses.as_ptr()
}
