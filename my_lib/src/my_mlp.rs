use rand::Rng;
use serde::{Serialize, Deserialize};
use crate::utils::reshape2D;

#[derive(Serialize, Deserialize)]
#[no_mangle]
pub struct MyMLP {
    d: Vec<i32>,
    w: Vec<Vec<Vec<f64>>>,
    l: usize,
    x: Vec<Vec<f64>>, 
    deltas: Vec<Vec<f64>>,
    train_losses: Vec<f64>,
    test_losses: Vec<f64>,
}

impl MyMLP {
    pub fn new(npl: &Vec<i32>) -> MyMLP {
        let mut mlp = MyMLP {
            d: npl.clone(),
            w: vec![],
            l: npl.len() - 1,
            x: vec![],
            deltas: vec![],
            train_losses: vec![],
            test_losses: vec![],
        };

        for l in 0..=mlp.l {
            mlp.w.push(Vec::new());

            if l == 0 {
                continue;
            }

            for _i in 0..(mlp.d[l - 1] + 1) as usize {
                mlp.w[l].push(Vec::new());

                for _j in 0..=mlp.d[l] {
                    if _j == 0 {
                        mlp.w[l][_i].push(0.0);
                    } else {
                        let random_value: f64 = rand::thread_rng().gen();
                        mlp.w[l][_i].push(random_value * 2.0 - 1.0);
                    }
                }
            }
        }

        for l in 0..=mlp.l {
            mlp.x.push(Vec::new()); 
            mlp.deltas.push(Vec::new()); 

            for j in 0..=mlp.d[l] {
                mlp.deltas[l].push(0.0);

                if j == 0 {
                    mlp.x[l].push(1.0); 
                } else {
                    mlp.x[l].push(0.0);
                }
            }
        }
        mlp
    }

    pub fn propagate(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) {
        for j in 0..sample_inputs.len() {
            self.x[0][j + 1] = sample_inputs[j];
        }

        for l in 1..self.l + 1 {
            for j in 1..(self.d[l] as usize) + 1 {
                let mut total = 0.0;
                for i in 0..(self.d[l - 1] as usize) + 1 {
                    total += self.w[l][i][j] * self.x[l - 1][i];
                }
                if is_classification || l < self.l {
                    total = total.tanh();
                }
                self.x[l][j] = total;
            }
        }
    }

    pub fn predict(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification);
        let pred = self.x[self.l][1..].to_vec();
        pred
    }

    pub fn train(&mut self, 
        X_train: &Vec<Vec<f64>>, 
        y_train: &Vec<Vec<f64>>,
        X_test: &Vec<Vec<f64>>,
        y_test: &Vec<Vec<f64>>,
        alpha: f64, 
        nb_iter: usize, 
        is_classification: bool) {

        for epoch in 0..nb_iter {
            let mut epoch_train_loss = 0.0;
            let mut epoch_test_loss = 0.0;

            // Training phase
            for k in 0..X_train.len() {
                let sample_inputs = &X_train[k];
                let sample_expected_outputs = &y_train[k]; 

                self.propagate(sample_inputs, is_classification);

                let mut sample_loss = 0.0;

                for j in 1..(self.d[self.l] as usize) + 1 {
                    let error = self.x[self.l][j] - sample_expected_outputs[j - 1];
                    sample_loss += error.powi(2);
                    self.deltas[self.l][j] = error;

                    if is_classification {
                        self.deltas[self.l][j] *= (1.0 - self.x[self.l][j].powi(2));
                    }
                }
                epoch_train_loss += sample_loss / (self.d[self.l] as f64);

                for l in (2..self.l).rev() {
                    for i in 1..(self.d[l - 1] as usize) + 1 {
                        let mut total = 0.0;
                        for j in 1..(self.d[l] as usize) + 1 {
                            total += self.w[l][i][j] * self.x[l - 1][i];
                        }
                        total *= 1.0 - self.x[l - 1][i].powi(2);
                        self.deltas[l - 1][i] = total;
                    }
                }

                for l in 1..self.l + 1 {
                    for i in 0..(self.d[l - 1] as usize) + 1 {
                        for j in 1..(self.d[l] as usize) + 1 {
                            self.w[l][i][j] -= alpha * self.x[l - 1][i] * self.deltas[l][j];
                        }
                    }
                }
            }

            self.train_losses.push(epoch_train_loss / (X_train.len() as f64));

            // Testing phase
            for k in 0..X_test.len() {
                let sample_inputs = &X_test[k];
                let sample_expected_outputs = &y_test[k]; 

                self.propagate(sample_inputs, is_classification);

                let mut sample_loss = 0.0;

                for j in 1..(self.d[self.l] as usize) + 1 {
                    let error = self.x[self.l][j] - sample_expected_outputs[j - 1];
                    sample_loss += error.powi(2);
                }
                epoch_test_loss += sample_loss / (self.d[self.l] as f64);
            }

            self.test_losses.push(epoch_test_loss / (X_test.len() as f64));

            println!("Epoch {}: Train Loss = {}, Test Loss = {}", epoch, self.train_losses.last().unwrap(), self.test_losses.last().unwrap());
        }
    }
}

pub fn save(model: &MyMLP, path: &str) {
    let serialized = serde_json::to_string(model).unwrap();
    std::fs::write(path, serialized).unwrap();
}

pub fn load(path: &str) -> MyMLP {
    let data = std::fs::read_to_string(path).unwrap();
    serde_json::from_str(&data).unwrap()
}

#[no_mangle]
pub extern "C" fn create_MyMLP(p_npl: *const i32, npl_len: i32) -> *mut MyMLP {
    let npl = unsafe { std::slice::from_raw_parts(p_npl, npl_len as usize) }.to_vec();
    let model = MyMLP::new(&npl);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model
}

#[no_mangle]
pub extern "C" fn train_MyMLP(
                                p_model: *mut MyMLP,

                                p_X_train: *const f64, 
                                X_train_shape_0: i32, X_train_shape_1: i32,

                                p_y_train: *const f64, 
                                y_train_shape_0: i32, y_train_shape_1: i32,

                                p_X_test: *const f64, 
                                X_test_shape_0: i32, X_test_shape_1: i32,

                                p_y_test: *const f64, 
                                y_test_shape_0: i32, y_test_shape_1: i32,

                                alpha: f64, 
                                nb_iter: i32, 
                                is_classification: bool) {
    
    let mut model = unsafe { &mut *p_model };

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
    
    model.train(&X_train, &y_train, &X_test, &y_test, alpha, nb_iter as usize, is_classification);
}

#[no_mangle]
pub extern "C" fn predict_MyMLP(
                                p_model: *mut MyMLP,

                                p_input: *const f64, 
                                input_shape_0: i32, input_shape_1: i32,

                                is_classification: bool) -> *const f64 {

    let mut model = unsafe { &mut *p_model };

    let flatten_input = unsafe {
        std::slice::from_raw_parts(p_input, (input_shape_0 * input_shape_1) as usize)
    }.to_vec();
    let input = reshape2D(flatten_input, (input_shape_0 as usize, input_shape_1 as usize));

    let predictions = input.iter()
        .map(|sample| model.predict(sample, is_classification))
        .flatten()
        .collect::<Vec<_>>();

    let leaked_predictions = Vec::leak(predictions);
    leaked_predictions.as_ptr()
}

#[no_mangle]
pub extern "C" fn save_MyMLP(p_model: *mut MyMLP, path: *const u8, path_len: usize) {
    let model = unsafe { &mut *p_model };
    let path = unsafe { std::slice::from_raw_parts(path, path_len) };
    let path_str = std::str::from_utf8(path).unwrap();
    save(model, path_str);
}

#[no_mangle]
pub extern "C" fn load_MyMLP(path: *const u8, path_len: usize) -> *mut MyMLP {
    let path = unsafe { std::slice::from_raw_parts(path, path_len) };
    let path_str = std::str::from_utf8(path).unwrap();
    let model = load(path_str);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model
}

#[no_mangle]
pub extern "C" fn get_train_losses(p_model: *mut MyMLP) -> *const f64 {
    let model = unsafe { &*p_model };
    let train_losses = model.train_losses.clone();
    let leaked_train_losses = Box::leak(train_losses.into_boxed_slice());
    leaked_train_losses.as_ptr()
}

#[no_mangle]
pub extern "C" fn get_test_losses(p_model: *mut MyMLP) -> *const f64 {
    let model = unsafe { &*p_model };
    let test_losses = model.test_losses.clone();
    let leaked_test_losses = Box::leak(test_losses.into_boxed_slice());
    leaked_test_losses.as_ptr()
}