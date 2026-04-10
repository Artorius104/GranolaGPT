import cv2
import numpy as np
import ctypes

from lib_loader import load_my_lib

seed = 42

path = "/home/bao/GranolaGPT/my_lib/target/release/libmy_lib.so"

lib = load_my_lib(path)

def load_model(lib, model_path):
    # Charger le modèle
    cnn_ptr = lib.load_MyCNN(model_path.encode('utf-8'))
    if not cnn_ptr:
        raise ValueError("Failed to load the model")
    return cnn_ptr

def predict_with_captured_image(cnn_ptr, image):
    # Convertir l'image en type approprié pour C
    image = np.ascontiguousarray(image, dtype=np.double)
    num_samples = 1
    input_dim_0, input_dim_1, input_dim_2 = image.shape

    # Faire la prédiction
    predictions_ptr = lib.predict_MyCNN(
        cnn_ptr,
        image,
        num_samples,
        input_dim_0,
        input_dim_1,
        input_dim_2
    )

    # Convertir les prédictions en numpy array
    predictions = np.ctypeslib.as_array(predictions_ptr, shape=(num_samples,))

    return predictions

def capture_and_predict(lib, model_path):
    # Charger le modèle
    cnn_ptr = load_model(lib, model_path)

    # Essayer différents indices de caméra
    for i in range(5):
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            break
    else:
        print("Failed to open any camera")
        return

    # Définir les coordonnées du rectangle (x, y, largeur, hauteur)
    rect_x, rect_y, rect_w, rect_h = 200, 150, 200, 200

    while True:
        # Capture frame-by-frame
        ret, frame = cap.read()
        
        if not ret:
            print("Failed to grab frame")
            break

        # Recadre l'image pour ne conserver que ce qui est dans le rectangle
        cropped_frame = frame[rect_y:rect_y + rect_h, rect_x:rect_x + rect_w]

        # Redimensionne l'image recadrée en 48x48x3
        resized_frame = cv2.resize(cropped_frame, (48, 48))

        # Normaliser l'image
        normalized_frame = resized_frame / 255.0

        # Faire la prédiction
        predictions = predict_with_captured_image(cnn_ptr, normalized_frame)

        # Obtenir l'index de la prédiction maximale
        max_index = np.argmax(predictions)

        # Dessine un rectangle sur le frame
        cv2.rectangle(frame, (rect_x, rect_y), (rect_x + rect_w, rect_y + rect_h), (0, 255, 0), 2)

        # Affiche l'index de la prédiction maximale
        cv2.putText(frame, f'Prediction: {max_index}', (rect_x, rect_y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)

        # Affiche la vidéo en cours
        cv2.imshow('Webcam', frame)

        # Attends pour une touche d'entrée
        key = cv2.waitKey(1)
        if key == 27:  # 27 correspond à la touche Échap
            break

    # Libère la capture et ferme les fenêtres
    cap.release()
    cv2.destroyAllWindows()
    lib.destroy_MyCNN(cnn_ptr)

# Exemple d'utilisation
# Chemin vers le fichier du modèle sauvegardé
path_model = "../saved_models/CNN_models/2024-07-23_01-56-15/best_model/best_model.json"

# Capturer et prédire en temps réel
capture_and_predict(lib, path_model)
