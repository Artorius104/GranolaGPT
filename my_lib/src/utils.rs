use std::cmp::{max, min};
extern crate nalgebra;
use nalgebra::{DMatrix, DVector};

pub fn chunk_vector(input: Vec<f64>, chunk_size: usize) -> Vec<Vec<f64>> {
    let mut result: Vec<Vec<f64>> = Vec::new();
    let mut chunk: Vec<f64> = Vec::new();

    for (i, num) in input.iter().enumerate() {
        chunk.push(*num);
        if (i + 1) % chunk_size == 0 || i == input.len() - 1 {
            result.push(chunk.clone());
            chunk.clear();
        }
    }

    result
}

pub fn matrix_from_2d_vec(data: &Vec<Vec<f64>>) -> DMatrix<f64> {
    let rows = data.len();
    let cols = data[0].len();
    let mut matrix = DMatrix::zeros(rows, cols);
    for i in 0..rows {
        for j in 0..cols {
            matrix[(i, j)] = data[i][j];
        }
    }
    matrix
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

pub fn print_matrix(vec: &Vec<Vec<f64>>) {
    println!("[");
    for inner_vec in vec {
        print!("  [");
        for (i, val) in inner_vec.iter().enumerate() {
            if i != 0 {
                print!(", ");
            }
            print!("{:.2}", val); // Affiche les valeurs avec deux chiffres après la virgule
        }
        println!("],");
    }
    println!("]");
}

pub fn gauss_kernel(x: &Vec<f64>, c: &Vec<f64>, gamma: f64) -> f64 {
    let mut sum = 0.0;
    for i in 0..x.len() {
        sum += (x[i] - c[i]).powi(2);
    }

    (-sum / (2.0 * gamma.powi(2))).exp()
}


pub fn zeros_padding(input: Vec<Vec<f64>>, padding_height: usize, padding_width: usize) -> Vec<Vec<f64>> {
    let input_height = input.len();
    let input_width = input[0].len();
    let new_height = input_height + 2 * padding_height;
    let new_width = input_width + 2 * padding_width;

    let mut padded_input = vec![vec![0.0; new_width]; new_height];

    for i in 0..input_height {
        for j in 0..input_width {
            padded_input[i + padding_height][j + padding_width] = input[i][j];
        }
    }

    padded_input
}

pub fn correlate2d(input: Vec<Vec<f64>>, kernel: Vec<Vec<f64>>, padding:&str) -> Vec<Vec<f64>> {
    let mut input = input.clone();
    let input_height = input.len();
    let input_width = input[0].len();
    let kernel_size = kernel.len();

    let (output_height, output_width) = match padding {
        "valid" => (max(0, input_height - kernel_size + 1), max(0, input_width - kernel_size + 1)),
        "same" => {
            let padding_height = (kernel_size - 1) / 2;
            let padding_width = (kernel_size - 1) / 2;
            input = zeros_padding(input.clone(), padding_height, padding_width);
            (input_height, input_width)
        },
        "full" => {
            let padding_height = kernel_size - 1;
            let padding_width = kernel_size - 1;
            input = zeros_padding(input, padding_height, padding_width);
            (input_height + kernel_size - 1, input_width + kernel_size - 1)
        },
        _ => panic!("Not valid padding"),
    };
    
    let mut output = vec![vec![0.; output_width]; output_height];

    for y in 0..output.len() {
        for x in 0..output[0].len() {
            let mut sum = 0.0;
            for ky in 0..kernel_size {
                for kx in 0..kernel_size {
                    sum += input[y + ky][x + kx] * kernel[ky][kx];
                }
            }
            output[y][x] = sum;
            
        }
    }

    output
}

pub fn reshape(input: Vec<f64>, reshape_shape: (usize, usize, usize, usize)) -> Vec<Vec<Vec<Vec<f64>>>> {
    let (n_images, image_depth, image_height, image_width) = reshape_shape;
    let mut reshaped: Vec<Vec<Vec<Vec<f64>>>> = Vec::with_capacity(n_images);

    let mut index = 0;
    for _ in 0..n_images {
        let mut image_depth_vec: Vec<Vec<Vec<f64>>> = Vec::with_capacity(image_depth);
        for _ in 0..image_depth {
            let mut image_height_vec: Vec<Vec<f64>> = Vec::with_capacity(image_height);
            for _ in 0..image_height {
                let mut image_width_vec: Vec<f64> = Vec::with_capacity(image_width);

                for _ in 0..image_width {
                    image_width_vec.push(input[index]);
                    index += 1;
                }

                image_height_vec.push(image_width_vec);
            }

            image_depth_vec.push(image_height_vec);
        }

        reshaped.push(image_depth_vec);
    }
    reshaped
}