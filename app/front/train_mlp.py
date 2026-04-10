"""
Script d'entraînement MLP rapide.
Utilise un sous-échantillon du dataset pour un entraînement en quelques secondes.

Lancement (depuis app/front/) :
    python train_mlp.py
"""
import os
import sys
import numpy as np
from PIL import Image

# Charge la librairie Rust via load_lib.py (même dossier)
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from load_lib import lib, seed

# ─── Paramètres ───────────────────────────────────────────────
IMAGE_SIZE = 32              # 32x32 pixels en RGB → 3072 features
MAX_TRAIN_PER_CLASS = 300    # sous-échantillon pour vitesse
MAX_TEST_PER_CLASS  = 75

_HERE        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH = os.path.join(_HERE, '..', '..', 'dataset_upgrade')
MODEL_PATH   = os.path.join(_HERE, 'best_model.json')

ALPHA   = 0.01
EPOCHS  = 100
# ──────────────────────────────────────────────────────────────


def load_split(split, max_per_class):
    X, y = [], []
    label_map = {'happy': 0, 'neutral': 1, 'sad': 2}
    for class_label, label_idx in label_map.items():
        folder = os.path.join(DATASET_PATH, split, class_label)
        if not os.path.exists(folder):
            print(f"  Dossier manquant : {folder}")
            continue
        files = sorted([
            f for f in os.listdir(folder)
            if f.lower().endswith(('.jpg', '.jpeg', '.png'))
        ])[:max_per_class]
        for filename in files:
            try:
                img = (Image.open(os.path.join(folder, filename))
                       .convert('RGB')
                       .resize((IMAGE_SIZE, IMAGE_SIZE)))
                X.append(np.array(img, dtype=np.float64).flatten() / 255.0)
                y.append(label_idx)
            except Exception as e:
                print(f"  Ignoré {filename} : {e}")
        print(f"  {split}/{class_label} : {len(files)} images chargées")
    return np.array(X, dtype=np.float64), np.array(y, dtype=int)


def to_onehot(y, num_classes=3):
    result = np.zeros((len(y), num_classes), dtype=np.float64)
    for i, lbl in enumerate(y):
        result[i, lbl] = 1.0
    return result


# ─── Chargement ───────────────────────────────────────────────
print("Chargement du dataset...")
X_train, y_train_labels = load_split('train', MAX_TRAIN_PER_CLASS)
X_test,  y_test_labels  = load_split('test',  MAX_TEST_PER_CLASS)

y_train = to_onehot(y_train_labels)
y_test  = to_onehot(y_test_labels)

# Mélange des données d'entraînement
rng = np.random.RandomState(seed)
idx = rng.permutation(len(X_train))
X_train, y_train, y_train_labels = X_train[idx], y_train[idx], y_train_labels[idx]

input_dim  = X_train.shape[1]   # 32*32*3 = 3072
output_dim = 3

print(f"\nTrain : {X_train.shape[0]} images  |  Test : {X_test.shape[0]} images")
print(f"Input dim : {input_dim}")

# ─── Construction du MLP ──────────────────────────────────────
layer_sizes = np.array([input_dim, 64, output_dim], dtype=np.int32)
num_layers  = len(layer_sizes)
mlp = lib.create_MyMLP(layer_sizes, num_layers, seed)
print(f"Architecture MLP : {list(layer_sizes)}")

# ─── Entraînement ─────────────────────────────────────────────
print(f"\nEntraînement : {EPOCHS} epochs, lr={ALPHA} ...")
lib.train_MyMLP(
    mlp,
    np.ascontiguousarray(X_train),
    np.ascontiguousarray(y_train),
    np.ascontiguousarray(X_test),
    np.ascontiguousarray(y_test),
    X_train.shape[0],
    X_test.shape[0],
    input_dim,
    output_dim,
    ALPHA,
    EPOCHS,
    True,   # is_classification
)

# ─── Évaluation ───────────────────────────────────────────────
predictions = []
for sample in X_test:
    sample_c = np.ascontiguousarray(sample, dtype=np.float64)
    out = lib.predict_MyMLP(mlp, sample_c, 1, input_dim, True)
    predictions.append([out[i] for i in range(output_dim)])

predictions = np.array(predictions)
predicted_classes = np.argmax(predictions, axis=1)
accuracy = np.mean(predicted_classes == y_test_labels)
print(f"\nPrécision finale (test) : {accuracy * 100:.2f}%")

# ─── Sauvegarde ───────────────────────────────────────────────
result = lib.save_MyMLP(mlp, MODEL_PATH.encode('utf-8'))
if result == 0:
    print(f"Modèle sauvegardé : {MODEL_PATH}")
else:
    print("ERREUR lors de la sauvegarde !")

lib.destroy_MyMLP(mlp)
