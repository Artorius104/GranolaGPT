import os
import sys
import io
from ctypes import *
import numpy as np
import matplotlib.pyplot as plt
import numpy.ctypeslib
from PIL import Image

from utils import *
from dataset import *

from lib_loader import load_my_lib

path = "../my_lib/target/release/deps/libmy_lib.so"

my_lib = load_my_lib(path)

seed = 42

# Exemple d'utilisation
path_to_dataset = '../dataset_upgrade'
image_size = 48  # Vous pouvez ajuster cette taille
X_train, y_train, X_test, y_test = load_dataset(path_to_dataset, image_size)
X_train, y_train = shuffle_dataset(X_train, y_train, seed)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

input_shape = X_train[0].shape 

X_train_flat = X_train.flatten()/255.
X_test_flat = X_test.flatten()/255.

y_train_flat = y_train.flatten()
y_test_flat = y_test.flatten()

# Créer un modèle linéaire
n_features = X_train.shape[1] * X_train.shape[2] * X_train.shape[3]
n_classes = y_train.shape[1]
model_ptr = my_lib.create_MyLinearModel(n_features, n_classes, seed)

# Entraîner le modèle
learning_rate = 0.00001
epochs = 10000
is_classification = True

my_lib.train_MyLinearModel(
    model_ptr,
    X_train.flatten().astype(np.double),
    y_train.flatten().astype(np.double),
    X_test.flatten().astype(np.double),
    y_test.flatten().astype(np.double),
    X_train.shape[0], X_test.shape[0],
    n_features, n_classes,
    learning_rate, epochs,
    int(is_classification)
)

my_lib.destroy_MyLinearModel(model_ptr)