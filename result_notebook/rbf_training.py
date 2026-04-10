import os
import sys
import io
import ctypes
import json
from ctypes import *
import numpy as np
import numpy.ctypeslib
from PIL import Image

from utils import plot_accuracies, plot_losses, load_json
from dataset import load_image, load_dataset, shuffle_dataset, z_normalize, display_images

from lib_loader import load_my_lib

seed = 42

# Charger le dataset
image_size = 48  # Taille des images
dataset_path = '../dataset_upgrade'
X_train, y_train, X_test, y_test = load_dataset(dataset_path, image_size)
X_train, y_train = shuffle_dataset(X_train, y_train)
X_test, y_test = shuffle_dataset(X_test, y_test)

print(f"X_train shape: {X_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"y_test shape: {y_test.shape}")

display_images(X_train, 10)

path = "../my_lib/target/release/deps/libmy_lib.so"

lib = load_my_lib(path)

# Normaliser les données
X_train_normalized, X_test_normalized = z_normalize(X_train, X_test)

# Aplatir les images pour MLP
X_train_flat = X_train_normalized.reshape(X_train_normalized.shape[0], -1)
X_test_flat = X_train_normalized.reshape(X_train_normalized.shape[0], -1)

# Paramètres
num_samples, input_dim = X_train_flat.shape
output_dim = y_train.shape[1]
y_dim = 1
gamma = 1.0
k = len(X_train)
seed = 42

# Créer un modèle RBF
rbf_model = lib.create_MyRBF(X_train, k, input_dim, y_dim, gamma, seed)

# Entraîner le modèle RBF
lib.train_MyRBF(
    rbf_model,
    X_train_flat,
    y_train,
    X_test_flat,
    y_test,
    num_samples,
    num_samples,
    input_dim,
    output_dim,
    0.1,
    10,
    True
)