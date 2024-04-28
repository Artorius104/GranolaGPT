use rand::Rng;

mod utils;
use  crate::utils::chunk_vector;

#[no_mangle]
pub struct MyMLP {
    d: Vec<i32>,
    w: Vec<Vec<Vec<f64>>>,
    l: usize,
    x: Vec<Vec<f64>>,
    deltas: Vec<Vec<f64>>,
}

impl MyMLP {
    pub fn new(npl: &Vec<i32>) -> MyMLP {
        let mut mlp = MyMLP {
            d: npl.clone(),
            w: vec![],
            l: npl.len() - 1,
            x: vec![],
            deltas: vec![],
        };

        for l in 0..=mlp.l {
            mlp.w.push(Vec::new());

            if l == 0 {
                continue;
            }

            for _i in 0..=mlp.d[l - 1] as usize {
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
                mlp.deltas[l].push(0.0);use std::vec::Vec;
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

        for j in 0..sample_inputs.len(){
            self.x[0][j+1] = sample_inputs[j];
        }

        for l in 1..self.l+1{
            for j in 1..(self.d[l] as usize) + 1{
                let mut total = 0.0;
                for i in 0..(self.d[l-1] as usize) + 1{
                    total += self.w[l][i][j] * self.x[l-1][i];
                }
    
                if is_classification || l<self.l{
                    total = total.tanh();
                }

                self.x[l][j] = total;
            }
        }
    }

    pub fn predict(&mut self, sample_inputs: &Vec<f64>, is_classification: bool) -> Vec<f64> {
        self.propagate(sample_inputs, is_classification);
        self.x[self.l][1..].to_vec()
    }

    pub fn train(&mut self, 
                all_samples_inputs: &Vec<Vec<f64>>, 
                all_samples_expected_outputs: &Vec<Vec<f64>>, 
                alpha: f64, 
                nb_iter: usize, 
                is_classification: bool) -> Vec<f64>{
                    
        let mut losses:Vec<f64> = Vec::new();

        // let mut best_weights:Vec<Vec<Vec<f64>>> = self.w.clone();
        // let mut min_loss = f64::INFINITY;
        // let mut es = 1000;
        // let mut tmp = 0;

        for epochs in 0..nb_iter{
            // if(tmp == es){
            //     break;
            // }

            let k = rand::thread_rng().gen_range(0..all_samples_inputs.len());
            let sample_inputs = &all_samples_inputs[k];
            let sample_expected_outputs = &all_samples_expected_outputs[k];

            self.propagate(sample_inputs, is_classification);

            // Calculer la perte (MSE) pour cette époque
            let mut mse_loss = 0.0;         

            for j in 1..(self.d[self.l] as usize) +1{       
                let error = self.x[self.l][j] - sample_expected_outputs[j - 1];
                self.deltas[self.l][j] = error;

                mse_loss += error.powf(2.);

                if is_classification{
                    self.deltas[self.l][j] *= (1. - self.x[self.l][j].powf(2.));
                }
            }
            mse_loss /= self.d[self.l] as f64;

            //########################################//
            // if(mse_loss <= min_loss){
            //     tmp += 1;
            // }
            // else {
            //     best_weights = self.w.clone();
            //     tmp = es;
            // }
            if(epochs%1 == 0){
                losses.push(mse_loss);
    
            }
            //########################################//

            for l in (2..self.l).rev(){
                for i in 1..(self.d[l-1] as usize) + 1{
                    let mut total = 0.;
                    for j in 1..(self.d[l] as usize) + 1{
                        total += self.w[l][i][j] * self.x[l-1][i];
                    }
                    total *= 1. - self.x[l-1][i].powf(2.);
                    self.deltas[l-1][i] = total;
                }
            }

            for l in 1..self.l + 1{
                for i in 0..(self.d[l-1] as usize) + 1{
                    for j in 1..(self.d[l] as usize) + 1{
                        self.w[l][i][j] -= alpha * self.x[l-1][i] * self.deltas[l][j];
                    }
                }
            }

        }
        // self.w = best_weights.clone();
        losses
    }
}

#[no_mangle]
pub extern "C" fn create_MyMLP(arr: *const i32, arr_size: i32) -> *mut MyMLP {
    let nlp = unsafe{ 
        std::slice::from_raw_parts(arr, arr_size as usize)
    };

    let model = MyMLP::new(&nlp.to_vec());

    let boxed_model = Box::new(model);
    let leaked_boxed_model = Box::leak(boxed_model);
    leaked_boxed_model
}

#[no_mangle]
pub extern "C" fn train_MyMLP(
                            p_model: *mut MyMLP,

                            p_X:*const f64,
                            X_chunk: i32,

                            p_y:*const f64,
                            y_chunk : i32,

                            n_samples: i32, 
                            alpha: f64, 
                            nb_iter: i32, 
                            is_classification: bool) -> *const f64{

    let model = unsafe {&mut *p_model};

    let X_flatten = unsafe {
        {std::slice::from_raw_parts(p_X, (X_chunk * n_samples) as usize)}
    };
    let X = chunk_vector(X_flatten.to_vec(), X_chunk as usize);

    let y_flatten = unsafe {
        {std::slice::from_raw_parts(p_y, (y_chunk * n_samples) as usize)}
    };
    let y = chunk_vector(y_flatten.to_vec(), y_chunk as usize);


    let losses = model.train(&X, &y, alpha, nb_iter as usize, is_classification);

    let leaked_losses = Vec::leak(losses.to_vec());
    leaked_losses.as_ptr()
}

#[no_mangle]
pub extern "C" fn predict_MyMLP(p_model:*mut MyMLP, 
                                is_classification: bool, 
                                p_samples:*const f64, 
                                samples_chunk:i32, 
                                n_samples:i32) -> *const f64{
    let mut predictions:Vec<f64> = vec![];
    let model = unsafe{&mut *p_model};

    let samples_flatten = unsafe {
        std::slice::from_raw_parts(p_samples, (samples_chunk * n_samples) as usize)
    };

    let samples = chunk_vector(samples_flatten.to_vec(), samples_chunk as usize);

    for sample in samples{
        let sample_prediction = model.predict(&sample, is_classification);
        for value in sample_prediction{
            predictions.push(value);
        }
    }

    let leaked_predictions = Vec::leak(predictions);
    leaked_predictions.as_ptr()
}