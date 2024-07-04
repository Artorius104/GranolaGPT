import os
import requests


def folders_management(source: str):
    folder_path = os.path.join(os.getcwd(), source)
    if not os.path.exists(folder_path):
        os.makedirs(folder_path)

    return folder_path


def download_image(url, save_path):
    if os.path.exists(save_path):
        print(f"L'image existe déjà : {save_path}")
        return

    response = requests.get(url)
    if response.status_code == 200:
        with open(save_path, 'wb') as file:
            file.write(response.content)
        print(f'Image téléchargée avec succès : {save_path}')
    else:
        print(f'Échec du téléchargement de l\'image : {url}')
