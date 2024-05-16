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