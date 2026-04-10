import os
import sys
import io
import ctypes
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib
from PIL import Image
from dataset import load_image, load_dataset, shuffle_dataset, z_normalize, display_images

from lib_loader import load_my_lib

seed = 42

path = "../my_lib/target/release/deps/libmy_lib.so"

lib = load_my_lib(path)

image_size = 48

dataset_path = '../dataset_upgrade'
X_train, y_train, X_test, y_test = load_dataset(dataset_path, image_size)
X_train, y_train = shuffle_dataset(X_train, y_train)
X_test, y_test = shuffle_dataset(X_test, y_test, seed)

X_train, X_test = z_normalize(X_train, X_test)

X_train_flat = X_train.reshape(X_train.shape[0], -1)
X_test_flat = X_test.reshape(X_test.shape[0], -1)

input_dim = X_train_flat.shape[1]
output_dim = y_train.shape[1]

########################################
layer_sizes = np.array([input_dim, 4, 8, output_dim], dtype=np.int32)
########################################

num_layers = len(layer_sizes)

mlp = lib.create_MyMLP(layer_sizes, num_layers, seed)

lib.summary_MyMLP(mlp)

alpha = 0.001    
epochs = 1000000
is_classification = True

print("\n")

lib.train_MyMLP(
    mlp,
    X_train_flat,
    y_train,
    X_test_flat,
    y_test,
    X_train_flat.shape[0],
    X_test_flat.shape[0],
    X_train_flat.shape[1],
    y_train.shape[1],
    alpha,
    epochs,
    is_classification
)

predictions = []
for sample in X_test_flat:
    sample = np.array(sample, dtype=np.float64)
    output = lib.predict_MyMLP(mlp, sample, 1, sample.size, is_classification)
    prediction = [output[i] for i in range(y_train.shape[1])]
    predictions.append(prediction)

predictions = np.array(predictions)

predicted_classes = np.argmax(predictions, axis=1)
true_classes = np.argmax(y_test, axis=1)

accuracy = np.mean(predicted_classes == true_classes)
print(f'Accuracy: {accuracy * 100:.2f}%')

for i in range(10):
    print(f"True: {true_classes[i]}, Predicted: {predicted_classes[i]}")

lib.destroy_MyMLP(mlp)