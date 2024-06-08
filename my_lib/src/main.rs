mod my_rbf_rb;
use crate::my_rbf_rb::MyRBF_rb;

mod my_rbf;
use crate::my_rbf::MyRBF;

mod my_linear_regression;
use crate::my_linear_regression::MyLinearRegression;

mod utils;

fn print_matrix(vec: &Vec<Vec<f64>>) {
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


fn main() {
    // Exemple de données
    let X_train: Vec<Vec<f64>> = vec![
        vec![1.0, 1.0],
        vec![3.0, 2.0],
        vec![1.0, 3.0],
        vec![2.0, 7.0],
        vec![3.0, 7.0],
    ];

    // Définition de y_train
    let y_train: Vec<Vec<f64>> = vec![
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![1.0, 0.0],
        vec![0.0, 1.0],
        vec![0.0, 1.0],
    ];

    let mut model = MyLinearRegression::new();
    model.train(X_train.clone(), y_train.clone());

    let y_pred = model.predict(X_train);

    println!("Predictions: {:?}", y_pred);
}