use crate::utils::{correlate2d};

use serde::{Serialize, Deserialize};
use std::fs::File;
use std::io::{self, Read, Write};
use std::f64::consts::E;
use std::ffi::CStr;
use std::os::raw::c_char;
use rand::Rng;


//********************************* LAYER ***************************************************************************************************

#[derive(Serialize, Deserialize)]
pub enum Layer {
    ConvLayer(ConvLayer),
    FlattenLayer(FlattenLayer),
    ActivationLayer(ActivationLayer),
    DenseLayer(DenseLayer),
    MaxPoolingLayer(MaxPoolingLayer)
}

impl Layer {
    pub fn clone(&self) -> Layer {
        match self {
            Layer::ConvLayer(conv_layer) => Layer::ConvLayer(conv_layer.clone()),
            Layer::FlattenLayer(flatten_layer) => Layer::FlattenLayer(flatten_layer.clone()),
            Layer::ActivationLayer(activation_layer) => Layer::ActivationLayer(activation_layer.clone()),
            Layer::DenseLayer(dense_layer) => Layer::DenseLayer(dense_layer.clone()),
            Layer::MaxPoolingLayer(maxpooling_layer) => Layer::MaxPoolingLayer(maxpooling_layer.clone()),
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        match self{
            Layer::ConvLayer(conv_layer) => conv_layer.forward(input),
            Layer::FlattenLayer(flatten_layer) => flatten_layer.forward(input),
            Layer::ActivationLayer(activation_layer) => activation_layer.forward(input),
            Layer::DenseLayer(dense_layer) => dense_layer.forward(input),
            Layer::MaxPoolingLayer(maxpooling_layer) => maxpooling_layer.forward(input)
        }
    }

    pub fn backward(&mut self, output_gradient:Vec<Vec<Vec<f64>>>, lr:f64) -> Vec<Vec<Vec<f64>>> {
        match self{
            Layer::ConvLayer(conv_layer) => conv_layer.backward(output_gradient, lr),
            Layer::FlattenLayer(flatten_layer) => flatten_layer.backward(output_gradient),
            Layer::ActivationLayer(activation_layer) => activation_layer.backward(output_gradient),
            Layer::DenseLayer(dense_layer) => dense_layer.backward(output_gradient, lr),
            Layer::MaxPoolingLayer(maxpooling_layer) => maxpooling_layer.backward(output_gradient)
        }
    }

    pub fn print_type(&self) {
        match self {
            Layer::ConvLayer(_) => println!("Layer Type: Convolutional"),
            Layer::FlattenLayer(_) => println!("Layer Type: Flatten"),
            Layer::ActivationLayer(_) => println!("Layer Type: Activation"),
            Layer::DenseLayer(_) => println!("Layer Type: Dense"),
            Layer::MaxPoolingLayer(_) => println!("Layer Type: MaxPooling"),
        }
    }
}

#[no_mangle]
pub extern "C" fn create_ConvLayer(input_depth: i32, 
                                input_height:i32, 
                                input_width:i32, 
                                kernel_size: i32,
                                n_kernels:i32) -> *mut Layer {

    let conv_layer = Layer::ConvLayer(ConvLayer::new((input_depth as usize, 
                                                                        input_height as usize, 
                                                                        input_width as usize), 
                                                                        kernel_size as usize, 
                                                                        n_kernels as usize));

    let boxed_conv_layer = Box::new(conv_layer);
    let leaked_conv_layer = Box::leak(boxed_conv_layer);
    leaked_conv_layer 
}

#[no_mangle]
pub extern "C" fn create_ActivationLayer(p_activation_name: *const c_char) -> *mut Layer {

    let c_str = unsafe {
        assert!(!p_activation_name.is_null());
        CStr::from_ptr(p_activation_name)
    };
    
    // Convertir &CStr en &str
    let activation_name = c_str.to_str().unwrap();
    
    let activation_layer = Layer::ActivationLayer(ActivationLayer::new(activation_name));

    let boxed_activation_layer = Box::new(activation_layer);
    let leaked_activation_layer = Box::leak(boxed_activation_layer);
    leaked_activation_layer 
}

#[no_mangle]
pub extern "C" fn create_MaxPoolingLayer(pool_size:i32) -> *mut Layer {

    let maxpooling_layer = Layer::MaxPoolingLayer(MaxPoolingLayer::new(pool_size as usize));

    let boxed_maxpooling_layer = Box::new(maxpooling_layer);
    let leaked_maxpooling_layer = Box::leak(boxed_maxpooling_layer);
    leaked_maxpooling_layer 
}

#[no_mangle]
pub extern "C" fn create_FlattenLayer() -> *mut Layer {

    let flatten_layer = Layer::FlattenLayer(FlattenLayer::new());

    let boxed_flatten_layer = Box::new(flatten_layer);
    let leaked_flatten_layer = Box::leak(boxed_flatten_layer);
    leaked_flatten_layer 
}

#[no_mangle]
pub extern "C" fn create_DenseLayer(input_size: i32, 
                                    output_size:i32) -> *mut Layer {

    let dense_layer = Layer::DenseLayer(DenseLayer::new(input_size as usize, output_size as usize));

    let boxed_dense_layer = Box::new(dense_layer);
    let leaked_dense_layer = Box::leak(boxed_dense_layer);
    leaked_dense_layer 
}

//********************************* CONVOLUTION LAYER ***************************************************************************************************

#[derive(Serialize, Deserialize)]
pub struct ConvLayer {
    input: Vec<Vec<Vec<f64>>>,
    input_shape: (usize, usize, usize),
    kernel_size: usize,
    n_kernels: usize,
    kernels: Vec<Vec<Vec<Vec<f64>>>>,
    biases: Vec<Vec<Vec<f64>>>,
}

impl ConvLayer {
    pub fn new(input_shape: (usize, usize, usize), kernel_size: usize, n_kernels: usize) -> Self {
        let (input_depth, input_height, input_width) = input_shape;
        let kernels_shape = (n_kernels, input_depth, kernel_size, kernel_size);
        let output_shape = (n_kernels, input_height - kernel_size + 1, input_width - kernel_size + 1);
        
        let mut kernels = vec![vec![vec![vec![0.0; kernel_size]; kernel_size]; input_depth]; n_kernels];
        for i in 0..n_kernels {
            for j in 0..input_depth {
                for k in 0..kernel_size {
                    for l in 0..kernel_size {
                        kernels[i][j][k][l] = rand::random::<f64>();
                    }
                }
            }
        }

        let mut biases = vec![vec![vec![0.0; output_shape.2]; output_shape.1]; n_kernels];
        for i in 0..n_kernels {
            for j in 0..output_shape.1 {
                for k in 0..output_shape.2 {
                    biases[i][j][k] = rand::random::<f64>();
                }
            }
        }

        ConvLayer {
            input: vec![vec![vec![0.;input_width];input_height];input_depth],
            input_shape,
            kernel_size,
            n_kernels,
            kernels,
            biases,
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let (input_depth, input_height, input_width) = self.input_shape;
        let mut output = self.biases.clone();
        
        for i in 0..self.n_kernels {
            for j in 0..input_depth {
                let correlated = correlate2d(input[j].clone(), self.kernels[i][j].clone(), "valid");
                for y in 0..correlated.len() {
                    for x in 0..correlated[0].len() {
                        output[i][y][x] += correlated[y][x];
                    }
                }
            }
        }

        output
    }

    pub fn backward(&mut self, output_gradient: Vec<Vec<Vec<f64>>>, learning_rate: f64) -> Vec<Vec<Vec<f64>>> {
        let (input_depth, input_height, input_width) = self.input_shape;
        let mut kernels_gradient = vec![vec![vec![vec![0.0; self.kernel_size]; self.kernel_size]; input_depth]; self.n_kernels];
        let mut input_gradient = vec![vec![vec![0.0; input_width]; input_height]; input_depth];

        for i in 0..self.n_kernels {
            for j in 0..input_depth {
                kernels_gradient[i][j] = correlate2d(self.input[j].clone(), output_gradient[i].clone(), "valid");
                let convolved = correlate2d(output_gradient[i].clone(), self.kernels[i][j].clone(), "full");
                for y in 0..input_height {
                    for x in 0..input_width {
                        input_gradient[j][y][x] += convolved[y][x];
                    }
                }
            }
        }

        for i in 0..self.n_kernels {
            for j in 0..input_depth {
                for k in 0..self.kernel_size {
                    for l in 0..self.kernel_size {
                        self.kernels[i][j][k][l] -= learning_rate * kernels_gradient[i][j][k][l];
                    }
                }
            }
        }

        for i in 0..self.n_kernels {
            for j in 0..output_gradient[i].len() {
                for k in 0..output_gradient[i][j].len() {
                    self.biases[i][j][k] -= learning_rate * output_gradient[i][j][k];
                }
            }
        }

        input_gradient
    }

    pub fn clone(&self) -> Self {
        ConvLayer {
            input : self.input.clone(),
            input_shape: self.input_shape,
            kernel_size: self.kernel_size,
            n_kernels: self.n_kernels,
            kernels: self.kernels.clone(),
            biases: self.biases.clone(),
        }
    }
}

//********************************* ACTIVATION LAYER ***************************************************************************************************

#[derive(Serialize, Deserialize)]
pub struct ActivationLayer{
    input : Vec<Vec<Vec<f64>>>,
    input_shape : (usize, usize, usize),
    activation_name:String
}

impl ActivationLayer{
    pub fn clone(&self) -> ActivationLayer{
        ActivationLayer{
            input : self.input.clone(),
            input_shape : self.input_shape,
            activation_name : self.activation_name.clone()
        }
    }

    pub fn new(activation_name:&str) -> ActivationLayer{
        let (activation, activation_prime) = match activation_name {
            "tanh" => (tanh as fn(f64) -> f64, tanh_prime as fn(f64) -> f64),
            _ => panic!("Not an activation")
        };
        ActivationLayer{
            input : vec![],
            input_shape:(0,0,0),
            activation_name:String::from(activation_name)
            }
    }

    pub fn forward(&mut self, input:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let (activation, activation_prime) = match self.activation_name.as_str() {
            "tanh" => (tanh as fn(f64) -> f64, tanh_prime as fn(f64) -> f64),
            _ => panic!("Not an activation")
        };
        self.input = input;
        self.input_shape = (self.input.len(), self.input[0].len(), self.input[0][0].len());
        let (output_depth, output_height, output_width) = self.input_shape;

        let mut output = self.input.clone();
        for d in 0..output_depth{
            for h in 0..output_height{
                for w in 0..output_width{
                    output[d][h][w] = (activation)(output[d][h][w]);
                }
            }
        }

        // println!("Activation done! output=shape = {:?}", (output_depth, output_height, output_width));

        output
    }

    pub fn backward(&self, output_gradient:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let (activation, activation_prime) = match self.activation_name.as_str() {
            "tanh" => (tanh as fn(f64) -> f64, tanh_prime as fn(f64) -> f64),
            _ => panic!("Not an activation")
        };
        // println!("Backward Activation");
        let (output_depth, output_height, output_width) = self.input_shape;
        // println!("output_gradient_shape = {:?}", (output_gradient.len(), output_gradient[0].len(), output_gradient[0][0].len()));
        let mut input_gradient = output_gradient.clone();
        for d in 0..output_depth{
            for h in 0..output_height{
                for w in 0..output_width{
                    input_gradient[d][h][w] = output_gradient[d][h][w] * (activation_prime)(self.input[d][h][w]);
                }
            }
        }
        input_gradient
    }
}

fn tanh(x: f64) -> f64 {
    x.tanh()
}

fn tanh_prime(x: f64) -> f64 {
    1.0 - tanh(x).powi(2)
}


//********************************* MAXPOOLING LAYER ***************************************************************************************************

#[derive(Serialize, Deserialize)]
pub struct MaxPoolingLayer{
    pub pool_size : usize,
    pub input_shape : (usize, usize, usize),
    pub gradient_mapping : Vec<Vec<Vec<(usize,usize,usize)>>>
}

impl MaxPoolingLayer{
    pub fn clone(&self) -> MaxPoolingLayer{
        MaxPoolingLayer{
            pool_size : self.pool_size,
            input_shape : self.input_shape,
            gradient_mapping : self.gradient_mapping.clone()
        }
    }

    pub fn new(pool_size:usize) -> MaxPoolingLayer{
        MaxPoolingLayer{
            pool_size : pool_size,
            input_shape : (0,0,0),
            gradient_mapping : vec![]
        }
    }

    pub fn forward(&mut self, input:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let (input_depth, input_height, input_weight) = (input.len(), input[0].len(), input[0][0].len());
        self.input_shape = (input_depth, input_height, input_weight);
        let (output_depth, output_height, output_width)  = (input_depth, input_height-self.pool_size+1, input_weight-self.pool_size+1);

        let mut output = vec![vec![vec![0.;output_width];output_height];output_depth];
        let mut gradient_mapping = vec![vec![vec![(0,0,0);output_width];output_height];output_depth];
        for d in 0..output_depth{
            for h in 0..output_height{
                for w in 0..output_width{
                    let mut max = f64::NEG_INFINITY;
                    let mut pos_max:(usize,usize,usize) = (d,0,0);
                    for i in 0..self.pool_size{
                        for j in 0..self.pool_size{
                            if input[d][h+i][w+j] > max{
                                max = input[d][h+i][w+j].clone();
                                pos_max = (d,h+i,w+j);
                            }
                        }
                    }
                    output[d][h][w] = max;
                    gradient_mapping[d][h][w] = pos_max;
                }
            }
        }
        self.gradient_mapping = gradient_mapping.clone();

        output
    }

    pub fn backward(&self, output_gradient:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let mut input_gradient:Vec<Vec<Vec<f64>>> = vec![vec![vec![0.;self.input_shape.2];self.input_shape.1];self.input_shape.0];
        let output_shape = (output_gradient.len(), output_gradient[0].len(), output_gradient[0][0].len());
        for d in 0..output_shape.0{
            for h in 0..output_shape.1{
                for w in 0..output_shape.2{
                    let pos = self.gradient_mapping[d][h][w].clone();
                    input_gradient[pos.0][pos.1][pos.2] = output_gradient[d][h][w];
                }
            }
        }
        input_gradient
    }
}


//********************************* FLATTEN LAYER ***************************************************************************************************

#[derive(Serialize, Deserialize)]
pub struct FlattenLayer{
    input_shape : (usize, usize, usize),
    output_size : usize
}

impl FlattenLayer{
    pub fn new() -> FlattenLayer{
        FlattenLayer{
            input_shape : (0, 0, 0),
            output_size : 0
        }
    }

    pub fn forward(&mut self, input:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let (input_depth, inputh_height, input_width) = (input.len(), input[0].len(), input[0][0].len());
        self.input_shape = (input_depth, inputh_height, input_width);
        self.output_size = input_depth * inputh_height * input_width;
        let mut flatten:Vec<f64> = vec![];
        for d in 0..input_depth{
            for h in 0..inputh_height{
                for w in 0..input_width{
                    flatten.push(input[d][h][w]);
                }
            }
        }
        let casted_flatten = vec![vec![flatten]]; 

        // println!("Flatten done! output=shape = {:?}", (casted_flatten.len(), casted_flatten[0].len(), casted_flatten[0][0].len()));
        casted_flatten
    }

    pub fn backward(&self, output_gradient:Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>>{
        let (input_depth, input_height, input_width) = self.input_shape;
        let mut result:Vec<Vec<Vec<f64>>> = vec![vec![vec![0.;input_width];input_height];input_depth];
        let i:usize = 0;
        for d in 0..input_depth{
            for h in 0..input_height{
                for w in 0..input_width{
                    result[d][h][w] = output_gradient[0][i][0];
                }
            }
        }
        result
    }

    pub fn clone(&self) -> FlattenLayer{
        FlattenLayer{
            input_shape : self.input_shape,
            output_size : self.output_size
        }
    }
}


#[derive(Serialize, Deserialize)]
pub struct DenseLayer {
    pub  weights: Vec<Vec<f64>>,
    pub bias: Vec<f64>,
    pub input: Vec<f64>,
}


//********************************* DENSE LAYER ***************************************************************************************************

impl DenseLayer {
    pub fn new(input_size: usize, output_size: usize) -> Self {
        let weights: Vec<Vec<f64>> = (0..output_size).map(|_| (0..input_size).map(|_| rand::thread_rng().gen_range(-1.0..=1.0)).collect()).collect();
        let bias: Vec<f64> = (0..output_size).map(|_| rand::thread_rng().gen_range(-1.0..=1.0)).collect();

        DenseLayer {
            weights,
            bias,
            input: vec![0.; input_size],
        }
    }

    pub fn forward(&mut self, input: Vec<Vec<Vec<f64>>>) -> Vec<Vec<Vec<f64>>> {
        let input = input[0][0].clone();
        if(input.len() != self.input.len()){
            println!("input shape {:?} does not match with {:?}", input.len(), self.input.len());
        }
        self.input = input.clone();
        let mut output: Vec<f64> = vec![0.; self.weights.len()];

        for i in 0..self.weights.len() {
            let mut sum = self.bias[i];
            for j in 0..self.input.len() {
                sum += self.weights[i][j] * self.input[j];
            }
            output[i] = sum;
        }

        let casted_output = vec![vec![output]];
        // println!("Dense done! output=shape = {:?}", (casted_output.len(), casted_output[0].len(), casted_output[0][0].len()));
        casted_output
    }

    pub fn backward(&mut self, output_gradient: Vec<Vec<Vec<f64>>>, learning_rate: f64) -> Vec<Vec<Vec<f64>>> {
        // println!("output_gradient = {:?}", output_gradient);
        let output_gradient = output_gradient[0][0].clone();
        let input_size = self.input.len();
        let output_size = output_gradient.len();

        let mut input_gradient: Vec<f64> = vec![0.0; input_size];
        let mut weights_gradient: Vec<Vec<f64>> = vec![vec![0.0; input_size]; output_size];
        let mut bias_gradient: Vec<f64> = vec![0.0; output_size];

        for i in 0..output_size {
            for j in 0..input_size {
                weights_gradient[i][j] = output_gradient[i] * self.input[j];
                input_gradient[j] += self.weights[i][j] * output_gradient[i];
            }
            bias_gradient[i] = output_gradient[i];
        }

        for i in 0..output_size {
            for j in 0..input_size {
                self.weights[i][j] -= learning_rate * weights_gradient[i][j];
            }
            self.bias[i] -= learning_rate * bias_gradient[i];
        }
        let casted_input_gradient = vec![vec![input_gradient]];

        // println!("Backward Dense done! output=shape = {:?}", (casted_input_gradient.len(), casted_input_gradient[0].len(), casted_input_gradient[0][0].len()));
        casted_input_gradient
    }

    pub fn clone(&self) -> DenseLayer{
        DenseLayer{
            weights : self.weights.clone(),
            bias : self.bias.clone(),
            input : self.input.clone()
        }
    }
}