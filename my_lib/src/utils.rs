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