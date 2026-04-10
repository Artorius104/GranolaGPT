# GranolaGPT

Application de reconnaissance d'émotion faciale (Happy / Neutral / Sad) en temps réel via webcam ou upload d'image.

Stack : librairie ML en **Rust** (MLP, CNN, RBF, modèle linéaire) + interface web **Dash Plotly** + interop Python/Rust via ctypes.

---

## Prérequis

- Python 3.10+
- [Rust / Cargo](https://rustup.rs/) (installé via `rustup`)
- Une webcam (optionnel — l'upload d'image fonctionne sans)
- Le dataset placé dans `dataset_upgrade/` (voir section Dataset)

---

## Installation

### 1. Cloner le repo

```bash
git clone <url-du-repo>
cd GranolaGPT
```

### 2. Installer Rust (si absent)

```bash
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh -s -- -y
source "$HOME/.cargo/env"
```

### 3. Compiler la librairie Rust

```bash
cd my_lib
cargo build --release
cd ..
```

Le fichier `my_lib/target/release/libmy_lib.so` est généré.

### 4. Créer l'environnement Python

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r app/requirements.txt
```

### 5. Entraîner le modèle MLP

Le modèle entraîné n'est pas versionné. Il faut le générer avant le premier lancement :

```bash
source venv/bin/activate
python app/front/train_mlp.py
```

Le script charge un sous-échantillon du dataset (300 images/classe), entraîne un MLP `[3072 → 64 → 3]` en ~100 epochs et sauvegarde le modèle dans `app/front/best_model.json`.

Pour améliorer la précision, éditer les paramètres en haut de [app/front/train_mlp.py](app/front/train_mlp.py) :

```python
MAX_TRAIN_PER_CLASS = 1000   # plus de données
EPOCHS  = 300                # plus d'epochs
```

### 6. Lancer l'application

```bash
source venv/bin/activate
export REACT_VERSION=18.2.0
python app/front/app.py
```

L'application est accessible sur **http://127.0.0.1:8050**

---

## Dataset

Le dataset doit être placé dans `dataset_upgrade/` avec la structure suivante :

```
dataset_upgrade/
├── train/
│   ├── happy/
│   ├── neutral/
│   └── sad/
└── test/
    ├── happy/
    ├── neutral/
    └── sad/
```

Sources utilisées :
- [CK+](https://www.kaggle.com/datasets/shawon10/ckplus)
- [Natural Human Face Images](https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition)
- [FER-2013](https://www.kaggle.com/datasets/msambare/fer2013)
- [CelebA](https://www.kaggle.com/datasets/jessicali9530/celeba-dataset)
- [Human Face Emotions](https://www.kaggle.com/datasets/sanidhyak/human-face-emotions)

---

## Utilisation

- **Flux vidéo** : la webcam s'affiche en temps réel avec l'émotion détectée en overlay (rectangle vert).
- **Prendre une photo** : capture le frame courant et affiche la prédiction.
- **Drag & Drop** : uploadez n'importe quelle image pour obtenir une prédiction.

---

## Notes Linux (webcam)

Sur les distributions modernes avec **PipeWire**, OpenCV 4.9 ne parvient pas à ouvrir la caméra. La version `>=4.10` est requise (déjà dans `requirements.txt`).

Si la caméra n'est pas détectée, vérifier que l'utilisateur est dans le groupe `video` :

```bash
groups   # doit afficher "video"
# si absent :
sudo usermod -aG video $USER
# puis se reconnecter ou :
newgrp video
```

---

## Structure du projet

```
GranolaGPT/
├── app/
│   ├── front/
│   │   ├── app.py          # Application Dash
│   │   ├── load_lib.py     # Wrapper ctypes de la librairie Rust
│   │   ├── train_mlp.py    # Script d'entraînement MLP (à lancer 1 fois)
│   │   └── best_model.json # Modèle entraîné (généré par train_mlp.py)
│   ├── requirements.txt
│   └── test_webcam.py
├── my_lib/                 # Librairie Rust (MLP, CNN, RBF, modèle linéaire)
│   ├── src/
│   └── Cargo.toml
├── dataset_upgrade/        # Dataset (non versionné)
└── result_notebook/        # Scripts d'entraînement et notebooks
```
