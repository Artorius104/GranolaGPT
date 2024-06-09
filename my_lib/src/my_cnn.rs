use std::ffi::CStr;
use std::os::raw::c_char;

use serde::{Serialize, Deserialize};
use serde_json;
use std::fs::File;
use std::io::{self, Read, Write};

use crate::layer::Layer;
use crate::utils::{reshape, reshape2D};

#[derive(Serialize, Deserialize)]
pub struct MyCNN{
    pub network:Vec<Layer>,
    pub loss_name:String,
    train_losses: Vec<f64>,
    test_losses: Vec<f64>
}

impl MyCNN{
    pub fn new(network:Vec<Layer>, loss_name:&str) -> MyCNN{
        let (loss, loss_prime) = match loss_name {
            "mse" => (mse as fn(Vec<f64>, Vec<f64>) -> f64, mse_prime as fn(Vec<f64>, Vec<f64>) -> Vec<f64>),
            _ => panic!("Not a loss function")
        };
        MyCNN{
            network : network,
            loss_name : String::from(loss_name),
            train_losses : vec![],
            test_losses: vec![]
        }
    }

    pub fn train(&mut self,  X_train: Vec<Vec<Vec<Vec<f64>>>>, y_train: Vec<Vec<f64>>, X_test: Vec<Vec<Vec<Vec<f64>>>>, y_test: Vec<Vec<f64>>, 
                        learning_rate: f64, 
                        epochs: usize) {
        let n_samples_train = X_train.len();
        let n_samples_test = X_test.len();
    
        let mut train_losses: Vec<f64> = vec![];
        let mut test_losses: Vec<f64> = vec![];
    
        for it in 0..epochs {
            let mut train_error = 0.;
    
            for (X_i, y_i) in X_train.iter().zip(y_train.iter()) {
                let input = X_i.clone();
                let prediction = self.predict(vec![input])[0][0][0].clone();
                let mut gradient: Vec<f64> = vec![];
    
                for (pred, true_value) in prediction.iter().zip(y_i.iter()) {
                    train_error += (*pred - *true_value).powf(2.);
    
                    let grad_pred = 2.0 * (pred - true_value) / y_i.len() as f64;
                    gradient.push(grad_pred);
                }
    
                let mut gradient = vec![vec![gradient]];
    
                for layer in self.network.iter_mut().rev() {
                    gradient = layer.backward(gradient.clone(), learning_rate);
                }
            }
            train_error /= n_samples_train as f64;
            train_losses.push(train_error);
    
            // Calcul de la loss sur le X_test
            let mut test_error = 0.;
            for (X_i, y_i) in X_test.iter().zip(y_test.iter()) {
                let input = X_i.clone();
                let prediction = self.predict(vec![input])[0][0][0].clone();
    
                for (pred, true_value) in prediction.iter().zip(y_i.iter()) {
                    test_error += (*pred - *true_value).powf(2.);
                }
            }
            test_error /= n_samples_test as f64;
            test_losses.push(test_error);
    
            println!("epoch {:?} => train_loss = {:?} || test_loss = {:?}", it, train_error, test_error);
        }
        self.train_losses = train_losses;
        self.test_losses = test_losses;
    }

    pub fn predict(&mut self, X:Vec<Vec<Vec<Vec<f64>>>>) ->Vec<Vec<Vec<Vec<f64>>>>{
        let mut predictions = Vec::new();

        for input in X.iter() {
            let mut output = input.clone();
            for layer in self.network.iter_mut() {
                output = layer.forward(output.clone());
            }
            // println!("final output =\n{:?}", output);

            predictions.push(output.clone());
        }
        predictions
    }
}

pub fn save(model: &MyCNN, path: &str) {
    let serialized = serde_json::to_string(model).unwrap();
    std::fs::write(path, serialized).unwrap();
}

pub fn load(path: &str) -> MyCNN {
    let data = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&data).unwrap()
}

fn mse(y_true: Vec<f64>, y_pred: Vec<f64>) -> f64 {
    let n = y_true.len();
    let mut sum = 0.0;
    
    for i in 0..n {
        sum += (y_true[i] - y_pred[i]).powi(2);
    }
    
    sum / n as f64
}

fn mse_prime(y_true: Vec<f64>, y_pred: Vec<f64>) -> Vec<f64> {
    let n = y_true.len();
    let mut gradient = vec![0.0; n];
    
    for i in 0..n {
        gradient[i] = 2.0 * (y_pred[i] - y_true[i]) / n as f64;
    }
    
    gradient
}

#[no_mangle]
pub extern "C" fn create_MyCNN(p_network : *const *mut Layer, 
                                network_size:i32, 
                                p_loss_name: *const c_char) -> *mut MyCNN{

    let tmp = unsafe {
        {std::slice::from_raw_parts(p_network, (network_size as usize))}
    };

    let mut network = Vec::<Layer>::new();
    for i in 0..network_size {
        let layer_ptr = unsafe {tmp[i as usize]};
        let layer_copy = unsafe { std::ptr::read(layer_ptr) };
        network.push(layer_copy);
    }

    for i in 0..network_size{
        network[i as usize].print_type();
    }

    let c_str = unsafe {
        assert!(!p_loss_name.is_null());
        CStr::from_ptr(p_loss_name)
    };

    let loss_name = c_str.to_str().unwrap();

    let model = MyCNN::new(network, loss_name);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model 
}

#[no_mangle]
pub extern "C" fn train_MyCNN(p_model: *mut MyCNN,
    
                            p_X_train:*const f64, 
                            n_images_train: i32, 
                            image_depth:i32, 
                            image_height:i32, 
                            image_width:i32,

                            p_y_train:*const f64,
                            y_dim:i32,

                            p_X_test:*const f64, 
                            n_images_test: i32, 
                        
                            p_y_test:*const f64,

                            learning_rate:f64,
                            epochs:i32
                            ){
    
    let mut model = unsafe {&mut *p_model};

    let X_train_flatten = unsafe {
        {std::slice::from_raw_parts(p_X_train, (n_images_train * image_depth * image_height * image_width) as usize)}
    }.to_vec();

    let y_train_flatten = unsafe {
        {std::slice::from_raw_parts(p_y_train, (n_images_train * y_dim) as usize)}
    }.to_vec();

    let X_test_flatten = unsafe {
        {std::slice::from_raw_parts(p_X_train, (n_images_test * image_depth * image_height * image_width) as usize)}
    }.to_vec();

    let y_test_flatten = unsafe {
        {std::slice::from_raw_parts(p_y_test, (n_images_test * y_dim) as usize)}
    }.to_vec();

    let X_train = reshape(X_train_flatten, (n_images_train as usize, image_depth as usize, image_height as usize, image_width as usize));
    let y_train = reshape2D(y_train_flatten, (n_images_train as usize, y_dim as usize));

    let X_test = reshape(X_test_flatten, (n_images_test as usize, image_depth as usize, image_height as usize, image_width as usize));
    let y_test = reshape2D(y_test_flatten, (n_images_test as usize, y_dim as usize));

    // println!("X =\n{:?}", X);
    // println!("y =\n{:?}\n", y);
    model.train(X_train, y_train, X_test, y_test, learning_rate, epochs as usize);

    // let leaked_losses = Vec::leak(losses.to_vec());
    // leaked_losses.as_ptr()
}


#[no_mangle]
pub extern "C" fn predict_MyCNN(p_model: *mut MyCNN, 

                                p_X:*const f64, 
                                n_images: i32, 
                                image_depth:i32,  image_height:i32, image_width:i32) -> *const f64 {
    let mut model = unsafe {&mut *p_model};

    let X_flatten = unsafe {
        {std::slice::from_raw_parts(p_X, (n_images * image_depth * image_height * image_width) as usize)}
    }.to_vec();

    let X = reshape(X_flatten, (n_images as usize, image_depth as usize, image_height as usize, image_width as usize));

    let predictions = model.predict(X);
    let flatten_predictions:Vec<f64> = predictions[0][0].clone().into_iter().flat_map(|v| v.into_iter()).collect();

    let leaked_predictions = Vec::leak(flatten_predictions);
    leaked_predictions.as_ptr() 
}

#[no_mangle]
pub extern "C" fn get_train_losses_MyCNN(p_model: *mut MyCNN) -> *const f64{
    let mut model = unsafe {&mut *p_model};
    let train_losses = model.train_losses.clone();

    let leaked_train_losses = Vec::leak(train_losses);
    leaked_train_losses.as_ptr() 
}

#[no_mangle]
pub extern "C" fn get_test_losses_MyCNN(p_model: *mut MyCNN) -> *const f64{
    let mut model = unsafe {&mut *p_model};
    let test_losses = model.test_losses.clone();

    let leaked_test_losses = Vec::leak(test_losses);
    leaked_test_losses.as_ptr() 
}

#[no_mangle]
pub extern "C" fn save_MyCNN(p_model: *mut MyCNN, path: *const c_char) {
    let model = unsafe { &*p_model };
    let c_str = unsafe {
        assert!(!path.is_null());
        CStr::from_ptr(path)
    };
    let path_str = c_str.to_str().unwrap();

    save(model, path_str);
}

#[no_mangle]
pub extern "C" fn load_MyCNN(path: *const c_char) -> *mut MyCNN {
    let c_str = unsafe {
        assert!(!path.is_null());
        CStr::from_ptr(path)
    };
    let path_str = c_str.to_str().unwrap();

    let model = load(path_str);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model
}