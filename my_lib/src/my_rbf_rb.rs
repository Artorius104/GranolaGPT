use rand::Rng;
use crate::utils::{reshape2D, gauss_kernel};

pub struct MyRBF_rb{
    weights : Vec<Vec<f64>>,
    biases : Vec<f64>,
    centers : Vec<Vec<f64>>,
    y_dim : usize,
    gamma:f64
}

impl MyRBF_rb{
    pub fn new(centers:Vec<Vec<f64>>, y_dim:usize, gamma:f64) -> Self{
        let (n_centers, n_features) = (centers.len(), centers[0].len());
        let mut weights= vec![vec![1.;n_centers];y_dim]; 
        let mut biases = vec![0.; y_dim];
        for i in 0..y_dim{
            for j in 0..n_centers{
                weights[i][j] = rand::thread_rng().gen_range(-1.0..=1.0);
            }
            biases[i] = rand::thread_rng().gen_range(-1.0..=1.0);
        }
        MyRBF_rb{
            weights : weights,
            biases : biases,
            centers : centers,
            y_dim : y_dim,
            gamma : gamma
        }
    }
    pub fn train(&mut self, X_train:Vec<Vec<f64>>, y_train:Vec<Vec<f64>>, learning_rate:f64, epochs:usize, is_classification:bool){
        let (n_centers, n_features) = (self.centers.len(), self.centers[0].len());
        for e in 0..epochs{
            for k in 0..X_train.len(){
                let mut hidden_output = vec![0.;n_centers];
                for c in 0..n_centers{
                    hidden_output[c] = gauss_kernel(&X_train[k],&self.centers[c], self.gamma);
                }
    
                let mut pred = vec![0.;self.y_dim];
                for i in 0..self.y_dim{
                    for j in 0..n_centers{
                        pred[i] += self.weights[i][j] * hidden_output[j];
                    }
                    pred[i] += self.biases[i];
                }
                if is_classification{
                    for i in 0..self.y_dim{
                        if pred[i] < 0.{
                            pred[i] = -1.;
                        }
                        else {
                            pred[i] = 1.;
                        }
                    }
                }
                let mut error = vec![0.0; self.y_dim];
                for i in 0..self.y_dim {
                    error[i] = y_train[k][i] - pred[i];
                }

                for i in 0..self.y_dim {
                    for j in 0..n_centers {
                        self.weights[i][j] += learning_rate * error[i] * hidden_output[j];
                    }
                    self.biases[i] += learning_rate * error[i];
                }
            }

            let mut total_error = 0.0;
            for (x, y) in X_train.iter().zip(y_train.iter()) {
                let prediction = self.predict(vec![x.clone()], false)[0].clone();
                for i in 0..self.y_dim {
                    total_error += (y[i] - prediction[i]).powi(2);
                }
            }
            // println!("Epoch {}: Total Error = {}", e + 1, total_error);
        }

    }

    pub fn predict(&self, input:Vec<Vec<f64>>, is_classification:bool) -> Vec<Vec<f64>>{
        let (n_centers, n_features) = (self.centers.len(), self.centers[0].len());
        let mut predictions = vec![];
        for k in 0..input.len(){
            let mut hidden_output = vec![0.;n_centers];
            for c in 0..n_centers{
                hidden_output[c] = gauss_kernel(&input[k],&self.centers[c], self.gamma);
            }

            let mut pred = vec![0.;self.y_dim];
            for i in 0..self.y_dim{
                for j in 0..n_centers{
                    pred[i] += self.weights[i][j] * hidden_output[j];
                }
                pred[i] += self.biases[i];
            }

            if is_classification{
                for i in 0..self.y_dim{
                    if pred[i] < 0.{
                        pred[i] = -1.;
                    }
                    else {
                        pred[i] = 1.;
                    }
                }
            }
            predictions.push(pred);
        }
        predictions
    }
}

#[no_mangle]
pub extern "C" fn create_MyRBF_rb(p_X_train:*const f64, 
                                X_train_shape_0:i32, X_train_shape_1:i32,
                                y_dim:i32,
                                gamma:f64
                                ) -> *mut MyRBF_rb{

    let flatten_X_train = unsafe {
        {std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)}
    }.to_vec();

    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let model = MyRBF_rb::new(X_train, y_dim as usize, gamma);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model 
}

#[no_mangle]
pub extern "C" fn train_MyRBF_rb(p_model:*mut MyRBF_rb,

                                p_X_train:*const f64, 
                                X_train_shape_0:i32, X_train_shape_1:i32,

                                p_y_train:*const f64, 
                                y_train_shape_0:i32, y_train_shape_1:i32,

                                learning_rate:f64,

                                epochs:i32,

                                is_classification:bool){

    let mut model = unsafe {&mut *p_model};

    let flatten_X_train = unsafe {
        {std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)}
    }.to_vec();
    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let flatten_y_train = unsafe {
        {std::slice::from_raw_parts(p_y_train, (y_train_shape_0 * y_train_shape_1) as usize)}
    }.to_vec();
    let y_train = reshape2D(flatten_y_train, (y_train_shape_0 as usize, y_train_shape_1 as usize));

    model.train(X_train, y_train, learning_rate, epochs as usize, is_classification)
}
    
// pub fn predict(&self, input: Vec<Vec<f64>>, is_classification: bool) -> Vec<Vec<f64>> 
#[no_mangle]
pub extern "C" fn predict_MyRBF_rb(p_model:*mut MyRBF_rb,
                                    
                                p_input:*const f64, 
                                input_shape_0:i32, input_shape_1:i32,
                                
                                is_classification:bool) -> *const f64{

    let mut model = unsafe {&mut *p_model};

    let flatten_input = unsafe {
        {std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)}
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = model.predict(input, is_classification);

    let mut flatten_pred = vec![];
    for i in 0..predictions.len(){
        for j in 0..predictions[0].len(){
            flatten_pred.push(predictions[i][j].clone());
        }
    }

    let leaked_predictions = Vec::leak(flatten_pred);
    leaked_predictions.as_ptr()
}