import os
import random
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from pathlib import Path
from PIL import Image
from typing import List, Tuple, Optional

from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.utils import to_categorical

def load_image(image_path, image_size):
    img = load_img(image_path, target_size=(image_size, image_size))
    img_array = img_to_array(img)
    return img_array

def load_images_from_folder(folder, label, image_size):
    images = []
    labels = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if img_path.endswith('.jpg') or img_path.endswith('.png'):
            img_array = load_image(img_path, image_size)
            images.append(img_array)
            labels.append(label)
    return images, labels

def load_dataset(path, image_size):
    X_train = []
    y_train = []
    X_test = []
    y_test = []

    label_map = {'happy': 0, 'neutral': 1, 'sad': 2}

    # Chargement des images de l'ensemble d'entraînement
    for class_label in ['happy', 'neutral', 'sad']:
        folder_path = os.path.join(path, 'train', class_label)
        images, labels = load_images_from_folder(folder_path, label_map[class_label], image_size)
        X_train.extend(images)
        y_train.extend(labels)

    # Chargement des images de l'ensemble de test
    for class_label in ['happy', 'neutral', 'sad']:
        folder_path = os.path.join(path, 'test', class_label)
        images, labels = load_images_from_folder(folder_path, label_map[class_label], image_size)
        X_test.extend(images)
        y_test.extend(labels)

    X_train = np.array(X_train, dtype=np.float64)
    y_train = np.array(y_train, dtype=np.float64)
    X_test = np.array(X_test, dtype=np.float64)
    y_test = np.array(y_test, dtype=np.float64)

    # One-hot encoding des étiquettes
    y_train = to_categorical(y_train, num_classes=3)
    y_test = to_categorical(y_test, num_classes=3)

    return X_train, y_train, X_test, y_test

def shuffle_dataset(
                    X: np.ndarray,
                    y: np.ndarray,
                    seed: Optional[int] = None
                ):

    if seed is not None:
        np.random.seed(seed)

    # Shuffle the data
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)
    X_shuffled = X[indices]
    y_shuffled = y[indices]

    return X_shuffled, y_shuffled

def display_images(X, n):
    plt.figure(figsize=(10, 10))
    for i in range(n):
        ax = plt.subplot(1, n, i + 1)
        plt.imshow(X[i].astype('uint8'))
        plt.axis("off")
    plt.show()

def z_normalize(X_train, X_test):
    mean = np.mean(X_train, axis=(0, 1, 2), keepdims=True)
    std = np.std(X_train, axis=(0, 1, 2), keepdims=True)
    X_train_normalized = (X_train - mean) / std
    X_test_normalized = (X_test - mean) / std
    
    return X_train_normalized, X_test_normalized