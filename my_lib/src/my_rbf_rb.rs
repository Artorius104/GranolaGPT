use rand::Rng;

pub struct MyRBF_rb {
    hidden_weights: Vec<Vec<f64>>, // Poids de la couche cachée
    output_weights: Vec<Vec<f64>>, // Poids de la couche de sortie
    output_bias: Vec<f64>, // Biais de la couche de sortie
    centers: Vec<Vec<f64>>, // Centres des neurones RBF
    gamma: f64, // gamma
}

impl MyRBF_rb {
    pub fn new(centers: Vec<Vec<f64>>, n_hidden_neurons: usize, n_output_neurons: usize, gamma: f64) -> Self {
        let (n_samples, n_features) = (centers.len(), centers[0].len());

        let mut hidden_weights = vec![vec![0.; n_samples]; n_hidden_neurons];
        for i in 0..n_hidden_neurons {
            for j in 0..n_samples {
                hidden_weights[i][j] = rand::thread_rng().gen_range(-1.0..=1.0);
            }
        }

        let mut output_weights = vec![vec![0.; n_hidden_neurons]; n_output_neurons];
        for i in 0..n_output_neurons {
            for j in 0..n_hidden_neurons {
                output_weights[i][j] = rand::thread_rng().gen_range(-1.0..=1.0);
            }
        }

        let mut output_bias = vec![0.; n_output_neurons];
        for i in 0..n_output_neurons {
            output_bias[i] = rand::thread_rng().gen_range(-1.0..=1.0);
        }

        MyRBF_rb {
            hidden_weights: hidden_weights,
            output_weights: output_weights,
            output_bias: output_bias,
            centers: centers.clone(),
            gamma: gamma,
        }
    }

    pub fn train(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, learning_rate: f64, epochs: usize, is_classification: bool) {
        let n_hidden_neurons = self.hidden_weights.len();
        let n_output_neurons = self.output_weights.len();

        for epoch in 0..epochs {
            for (x, y) in X_train.iter().zip(y_train.iter()) {
                let mut hidden_output: Vec<f64> = vec![0.0; n_hidden_neurons];
                for i in 0..n_hidden_neurons {
                    let mut res: f64 = 0.0;
                    for j in 0..self.centers.len() {
                        let gauss = gauss_kernel(x, &self.centers[j], self.gamma);
                        res += gauss * self.hidden_weights[i][j];
                    }
                    hidden_output[i] = res;
                }

                let mut output = vec![0.0; n_output_neurons];
                for i in 0..n_output_neurons {
                    for j in 0..n_hidden_neurons {
                        output[i] += self.output_weights[i][j] * hidden_output[j];
                    }
                    output[i] += self.output_bias[i];
                }

                let mut error = vec![0.0; n_output_neurons];
                for i in 0..n_output_neurons {
                    error[i] = y[i] - output[i];
                }
                for i in 0..n_output_neurons {
                    for j in 0..n_hidden_neurons {
                        self.output_weights[i][j] += learning_rate * error[i] * hidden_output[j];
                    }
                    self.output_bias[i] += learning_rate * error[i];
                }
            }

            let mut total_error = 0.0;
            for (x, y) in X_train.iter().zip(y_train.iter()) {
                let prediction = self.predict(vec![x.clone()], false)[0].clone();
                for i in 0..n_output_neurons {
                    total_error += (y[i] - prediction[i]).powi(2);
                }
            }
            println!("Epoch {}: Total Error = {}", epoch + 1, total_error);
        }
    }

    pub fn predict(&self, input: Vec<Vec<f64>>, is_classification: bool) -> Vec<Vec<f64>> {
        let n_hidden_neurons = self.hidden_weights.len();
        let n_output_neurons = self.output_weights.len();
        let n_samples = input.len();
        let mut predictions = vec![vec![0.0; n_output_neurons]; n_samples];

        for k in 0..n_samples {
            let mut hidden_output: Vec<f64> = vec![0.0; n_hidden_neurons];
            let single_sample = &input[k];
            for i in 0..n_hidden_neurons {
                let mut res: f64 = 0.0;
                for j in 0..self.centers.len() {
                    let gauss = gauss_kernel(&single_sample, &self.centers[j], self.gamma);
                    res += gauss * self.hidden_weights[i][j];
                }
                hidden_output[i] = res;
            }

            let mut pred = vec![0.0; n_output_neurons];
            for i in 0..n_output_neurons {
                for j in 0..n_hidden_neurons {
                    pred[i] += self.output_weights[i][j] * hidden_output[j];
                }
                pred[i] += self.output_bias[i];
            }

            predictions[k] = pred;

            if is_classification == true {
                let mut max_index = 0;
                let mut max_value = f64::NEG_INFINITY;
                for i in 0..n_output_neurons {
                    if predictions[k][i] > max_value {
                        max_index = i;
                        max_value = predictions[k][i];
                    }
                }
                pred = vec![0.0; n_output_neurons];
                pred[max_index] = 1.0;
                predictions[k] = pred;
            }
        }
        predictions
    }
}

fn gauss_kernel(x: &Vec<f64>, c: &Vec<f64>, sigma: f64) -> f64 {
    if x.len() != c.len() {
        panic!("Les vecteurs x et c doivent avoir la même longueur");
    }
    
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - c[i]).powi(2);
    }

    (-sum / (2.0 * sigma.powi(2))).exp()
}

//pub fn new(X_train:Vec<Vec<f64>>, n_hidden_neurons:usize, n_output_neurons:usize, gamma:f64)

#[no_mangle]
pub extern "C" fn create_MyRBF_rb(p_X_train:*const f64, 
                                X_train_shape_0:i32, X_train_shape_1:i32,
                                n_hidden_neurons:i32,
                                n_output_neurons:i32,
                                gamma:f64
                                ) -> *mut MyRBF_rb{

    let flatten_X_train = unsafe {
        {std::slice::from_raw_parts(p_X_train, (X_train_shape_0 * X_train_shape_1) as usize)}
    }.to_vec();

    let X_train = reshape2D(flatten_X_train, (X_train_shape_0 as usize, X_train_shape_1 as usize));

    let model = MyRBF_rb::new(X_train, n_hidden_neurons as usize, n_output_neurons as usize, gamma);
    let boxed_model = Box::new(model);
    let leaked_model = Box::leak(boxed_model);
    leaked_model 
}

//pub fn train(&mut self, X_train: Vec<Vec<f64>>, y_train: Vec<Vec<f64>>, gamma: f64, epochs: usize, is_classification: bool)
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


pub fn reshape2D(input: Vec<f64>, reshape_shape: (usize, usize)) -> Vec<Vec<f64>> {
    let (rows, cols) = reshape_shape;
    let total_elements = rows * cols;
    
    // Vérifie si la taille du vecteur d'entrée correspond aux dimensions souhaitées
    if input.len() != total_elements {
        panic!("La taille du vecteur d'entrée ne correspond pas aux dimensions souhaitées");
    }

    // Crée un nouveau vecteur 2D pour stocker le résultat
    let mut output = vec![vec![0.0; cols]; rows];

    for i in 0..rows {
        for j in 0..cols {
            output[i][j] = input[i * cols + j];
        }
    }

    output
}