import os
import shutil
from sklearn.model_selection import train_test_split
import random

def split_dataset(dataset_path, train_ratio=0.8):
    emotions = ['sad', 'neutral', 'happy']
    train_path = os.path.join(dataset_path, 'train')
    test_path = os.path.join(dataset_path, 'test')

    # Créer les dossiers train et test avec les sous-dossiers pour les émotions
    for emotion in emotions:
        os.makedirs(os.path.join(train_path, emotion), exist_ok=True)
        os.makedirs(os.path.join(test_path, emotion), exist_ok=True)

    # Parcourir chaque dossier d'émotion
    for emotion in emotions:
        emotion_path = os.path.join(dataset_path, emotion)
        images = os.listdir(emotion_path)
        
        # Mélanger les images
        random.shuffle(images)
        
        # Séparer les images en train et test
        train_images, test_images = train_test_split(images, train_size=train_ratio, random_state=42)
        
        # Copier les images dans les dossiers appropriés
        for image in train_images:
            src = os.path.join(emotion_path, image)
            dst = os.path.join(train_path, emotion, image)
            shutil.copyfile(src, dst)
        
        for image in test_images:
            src = os.path.join(emotion_path, image)
            dst = os.path.join(test_path, emotion, image)
            shutil.copyfile(src, dst)

    print("Dataset split completed.")

# Utilisation de la fonction
split_dataset('../dataset_upgrade')
