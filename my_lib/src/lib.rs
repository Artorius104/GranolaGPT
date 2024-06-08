mod utils;
use utils::{reshape2D};

mod my_mlp;
use my_mlp::{create_MyMLP, train_MyMLP, predict_MyMLP};

mod my_linear_regression;
use my_linear_regression::{create_MyLinearRegression, train_MyLinearRegression, predict_MyLinearRegression};

mod my_rbf_rb;
use my_rbf_rb::{create_MyRBF_rb, train_MyRBF_rb, predict_MyRBF_rb};

mod my_rbf;
use my_rbf::{create_MyRBF, train_MyRBF, predict_MyRBF};