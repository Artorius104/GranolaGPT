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