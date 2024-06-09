mod utils;
use utils::{reshape2D};

mod my_mlp;
use my_mlp::{create_MyMLP, train_MyMLP, predict_MyMLP, save_MyMLP, load_MyMLP};

mod my_linear_regression;
use my_linear_regression::{create_MyLinearRegression, train_MyLinearRegression, predict_MyLinearRegression};

mod my_linear_regression_rb;
use my_linear_regression_rb::{create_MyLinearRegression_rb,predict_MyLinearRegression_rb,train_MyLinearRegression_rb};

mod my_rbf;
use my_rbf::{create_MyRBF, train_MyRBF, predict_MyRBF};

mod my_rbf_rb;

mod layer;

mod my_cnn;