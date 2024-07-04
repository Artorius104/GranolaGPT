# GranolaGPT

## La Librairie
La librairie a été réalisé en Rust avec l'interop vers Python.

Les algorithmes réalisés (ou en cours de réalisation) sont les suivants :
  - Modèle linéaire
  - MLP
  - RBF
  - CNN

Pour build la librairie, l'utilisateur doit lancer la commande suivante :
```
cargo build --release  
```

## Le Dataset

Le dataset est composé des datasets suivants :
  - CK PLUS : https://www.kaggle.com/datasets/shawon10/ckplus
  - Natural Human Face Images for Emotion Recognition : https://www.kaggle.com/datasets/sudarshanvaidya/random-images-for-face-emotion-recognition
  - FER-2013 : https://www.kaggle.com/datasets/msambare/fer2013
  - CelebFaces Attributes (CelebA) Dataset : https://www.kaggle.com/datasets/jessicali9530/celeba-dataset
  - Human Face Emotions : https://www.kaggle.com/datasets/sanidhyak/human-face-emotions


## L'Application

L'application est réalisée avec Dash Plotly.
L'utilisateur peut utiliser sa webcam et cliquer sur le bouton "PRENDRE UNE PHOTO" pour capturer l'image.
Une image peut également être uploader via le bouton "DRAG & DROP".

L'image est ensuite traitée pour la stocker dans une variable correspondant à sa matrice de pixels.
La librairie est chargée puis l'image prise est envoyée au modèle.
La réponse du modèle est enfin affiché en bas de la page.


Pour mettre en place l'environnement, l'utilisateur doit d'abord se déplacer dans le dossier **/app** puis exécuter le **requirements.txt** à l'aide de la commande suivante :
```
pip install -r requirements.txt
```

Dash Plotly étant basé sur React, il est nécessaire de préciser sa version avec la commande suivante :
```
export REACT_VERSION=18.2.0
```

Pour vérifier que la webcam est utilisable, l'utilisateur peut exécuter le script **test_webcam.py** situé dans **/app**.
Si la réponse obtenue est : **Success: Camera opened**, alors la webcam est utilisée.
```
python test_webcam.py
```

Pour lancer l'application, l'utilisateur doit aller dans **/app/front** et lancer la commande suivante à partir de là :
```
python app.py
```

## Expérimentation

(Parler des problèmes d'apprentissage avec un dataset black&white)
(Parler des problèmes d'apprentissage avec les images de la classe très différentes des 2 autres)
(Parler de l'amélioration après le stripe des images happy et neutral)
(Scrapping des 2 autres émotions au cas où)
