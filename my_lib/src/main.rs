mod my_rbf_rb;
use crate::my_rbf_rb::MyRBF_rb;

mod my_rbf;
use crate::my_rbf::MyRBF;

mod my_linear_regression;
use crate::my_linear_regression::MyLinearRegression;

mod my_linear_regression_rb;
use crate::my_linear_regression_rb::MyLinearRegression_rb;

mod utils;

mod layer;

mod my_cnn;

fn print_matrix(vec: &Vec<Vec<f64>>) {
    println!("[");
    for inner_vec in vec {
        print!("  [");
        for (i, val) in inner_vec.iter().enumerate() {
            if i != 0 {
                print!(", ");
            }
            print!("{:.2}", val);
        }
        println!("],");
    }
    println!("]");
}

fn main() {
    let x_train = vec![
        vec![0.0, 0.0],
        vec![0.0, 1.0],
        vec![1.0, 0.0],
        vec![1.0, 1.0],
    ];
    let y_train = vec![
        vec![-1.0],
        vec![1.0],
        vec![1.0],
        vec![-1.0],
    ];

    let mut model = MyRBF::new(x_train.clone(), 0.01);
    model.train(x_train.clone(), y_train.clone());

    let predictions = model.predict(x_train, true);

    print_matrix(&predictions);
}