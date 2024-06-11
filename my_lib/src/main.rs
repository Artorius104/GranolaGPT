mod my_rbf_rb;
use crate::my_rbf_rb::MyRBF_rb;

mod my_rbf;
use crate::my_rbf::MyRBF;

mod my_linear_regression;
use crate::my_linear_regression::MyLinearRegression;

mod my_linear_model_rb;
use my_linear_model_rb::MyLinearModel_rb;


mod utils;
use crate::utils::{print_matrix, print_3d, generate_random_2d_tensor, generate_random_4d_tensor, reshape, reshape2D};

mod layer;
use crate::layer::{ConvLayer, ActivationLayer, MaxPoolingLayer,FlattenLayer,DenseLayer, Layer};

mod my_cnn;
use crate::my_cnn::{MyCNN, MyCNN_save, MyCNN_load};

fn main() {
    let shape = (100, 1, 28, 28);
    let X_train = generate_random_4d_tensor(shape);
    let X_test = generate_random_4d_tensor(shape);
    let y_train = generate_random_2d_tensor((2, 10));
    let y_test = generate_random_2d_tensor((2, 10));

    let network: Vec<Layer> = vec![
        Layer::ConvLayer(ConvLayer::new((1, 28, 28), 3, 3)),
        Layer::ActivationLayer(ActivationLayer::new("tanh")),   

        // Layer::MaxPoolingLayer(MaxPoolingLayer::new(2)),
        Layer::FlattenLayer(FlattenLayer::new()),

        Layer::DenseLayer(DenseLayer::new(2028, 10)),
        Layer::ActivationLayer(ActivationLayer::new("tanh")),
    ];

    let mut model = MyCNN::new(network, "mse");

    model.train(X_train, y_train, X_test, y_test, 0.1, 10);

    // Save and load the model
    MyCNN_save(&model, "my_cnn_model.json");
    let loaded_model = MyCNN_load("my_cnn_model.json");

    // Verify that the loaded model is the same
    println!("Train Losses: {:?}", loaded_model.train_losses);
    println!("Test Losses: {:?}", loaded_model.test_losses);
}