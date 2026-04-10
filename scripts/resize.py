from PIL import Image
import os

def resize_images_in_folder(folder_path, target_size=(151, 185)):
    # Vérifie si le dossier existe
    if not os.path.exists(folder_path):
        print(f"Le dossier {folder_path} n'existe pas.")
        return
    
    # Créé un dossier de sortie pour les images redimensionnées
    output_folder = os.path.join(folder_path, "resized")
    os.makedirs(output_folder, exist_ok=True)
    
    # Parcourt tous les fichiers du dossier
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        
        # Ouvre uniquement les fichiers d'image
        try:
            with Image.open(file_path) as img:
                # Redimensionne l'image
                img_resized = img.resize(target_size)
                
                # Sauvegarde l'image redimensionnée dans le dossier de sortie
                output_path = os.path.join(output_folder, filename)
                img_resized.save(output_path)
                
                print(f"{filename} a été redimensionnée et sauvegardée dans {output_folder}.")
        except IOError:
            print(f"{filename} n'est pas une image ou ne peut pas être ouverte.")

# Exemple d'utilisation
folder_path = "sad"
resize_images_in_folder(folder_path)
