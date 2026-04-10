import os
from PIL import Image
import json
import numpy as np
import matplotlib.pyplot as plt

def moving_average(data, window_size):
    """
    Calcule la moyenne mobile d'une série de données.
    
    :param data: Liste de valeurs pour lesquelles la moyenne mobile doit être calculée.
    :param window_size: Taille de la fenêtre pour le calcul de la moyenne mobile.
    :return: Liste des valeurs de la moyenne mobile.
    """
    return np.convolve(data, np.ones(window_size)/window_size, mode='valid')

# Nouvelle fonction pour afficher les courbes de précision avec la moyenne mobile
def plot_accuracies_moving_average(train_accuracies, test_accuracies, window_size):
    smoothed_train_accuracies = moving_average(train_accuracies, window_size)
    smoothed_test_accuracies = moving_average(test_accuracies, window_size)
    epochs = range(1, len(smoothed_train_accuracies) + 1)
    plt.plot(epochs, smoothed_train_accuracies, label='Training accuracy (MA)')
    plt.plot(epochs, smoothed_test_accuracies, label='Testing accuracy (MA)')
    plt.title('Training and Testing accuracy with Moving Average')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()

# Nouvelle fonction pour afficher les courbes de pertes avec la moyenne mobile
def plot_losses_moving_average(train_losses, test_losses, window_size):
    smoothed_train_losses = moving_average(train_losses, window_size)
    smoothed_test_losses = moving_average(test_losses, window_size)
    epochs = range(1, len(smoothed_train_losses) + 1)
    plt.plot(epochs, smoothed_train_losses, label='Training loss (MA)')
    plt.plot(epochs, smoothed_test_losses, label='Testing loss (MA)')
    plt.title('Training and Testing loss with Moving Average')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

def load_json(path):
    try:
        with open(path, 'r') as fichier:
            donnees = json.load(fichier)
        return donnees
    except FileNotFoundError:
        print(f"Erreur : le fichier {path} n'a pas été trouvé.")
    except json.JSONDecodeError:
        print(f"Erreur : le fichier {path} n'est pas un JSON valide.")
    except Exception as e:
        print(f"Une erreur s'est produite : {e}")

# Fonction pour afficher les courbes de précision
def plot_accuracies(train_accuracies, test_accuracies):
    epochs = range(1, len(train_accuracies) + 1)
    plt.plot(epochs, train_accuracies, label='Training accuracy')
    plt.plot(epochs, test_accuracies, label='Testing accuracy')
    plt.title('Training and Testing accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.show()
    
# Fonction pour afficher les courbes de pertes
def plot_losses(train_losses, test_losses):
    epochs = range(1, len(train_losses) + 1)
    plt.plot(epochs, train_losses, label='Training loss')
    plt.plot(epochs, test_losses, label='Testing loss')
    plt.title('Training and Testing loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
