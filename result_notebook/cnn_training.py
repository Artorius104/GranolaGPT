import os
import sys
import io
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib
from PIL import Image

from utils import plot_accuracies, plot_losses
from dataset import load_image, load_dataset, shuffle_dataset, z_normalize, display_images

from lib_loader import load_my_lib

path = "../my_lib/target/release/deps/libmy_lib.so"

my_lib = load_my_lib(path)

seed = 42


path_to_dataset = '../dataset_upgrade'
image_size = 48 
X_train, y_train, X_test, y_test = load_dataset(path_to_dataset, image_size)
X_train, y_train = shuffle_dataset(X_train, y_train, seed)
X_train, X_test = z_normalize(X_train, X_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

display_images(X_train, 10)

input_shape = X_train[0].shape 

X_train_flat = X_train.flatten()
X_test_flat = X_test.flatten()

y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()

model = my_lib.create_MyCNN(input_shape[0], input_shape[1], input_shape[2])

my_lib.add_conv_layer_MyCNN(model, 16, 3, 1, "Tanh".encode('utf-8'), 0.0, seed)
my_lib.add_maxpooling_layer_MyCNN(model, 2)

my_lib.add_conv_layer_MyCNN(model, 8, 3, 1, "Tanh".encode('utf-8'), 0.0, seed)
my_lib.add_maxpooling_layer_MyCNN(model, 2)

my_lib.add_dense_layer_MyCNN(model, 8, "Tanh".encode('utf-8'), 0.0, seed)
my_lib.add_dense_layer_MyCNN(model, 3, "Tanh".encode('utf-8'), 0.0, seed)

my_lib.set_optimizer_SGD(model, 0.01)

epochs = 100000
batch_size = 1
my_lib.train_MyCNN(
                model,
                X_train_flat, y_train_flat,
                X_test_flat, y_test_flat,
                X_train.shape[0],
                X_test.shape[0],  
                input_shape[0], input_shape[1], input_shape[2],
                y_train.shape[1],
                epochs,
                batch_size,
                seed
            )

my_lib.destroy_MyCNN(model)