import os
import requests
from requests.exceptions import HTTPError
import urllib.parse
import time

from utils import folders_management, download_image


# Paths env variables
absolute_path = os.path.dirname(__file__)
res_path = os.path.join(absolute_path, "../../../dataset")
API_KEY = {
    "pixabay": "44412475-410a83f69c39b1f5e6908d7f5",
}
queries = [
    "sad", "sadness", "sad human", "sad person", "sad man", "sad woman", "sad child", "sad boy", "sad girl",
    "sad portrait", "sad human portait", "sad person portrait", "sad man portrait", "sad woman portrait", "sad child portrait", "sad boy portrait", "sad girl portrait",
    "sad face", "sad human face", "sad person face", "sad man face", "sad woman face", "sad child face", "sad boy face", "sad girl face",
    "crying", "crying human", "crying person", "crying man", "crying woman", "crying child", "crying boy", "crying girl",
    "crying portrait", "crying human portrait", "crying person portrait", "crying man portrait", "crying woman portrait", "crying child portrait", "crying boy portrait", "crying girl portrait",
    "crying face", "crying human face", "crying person face", "crying man face", "crying woman face", "crying child face", "crying boy face", "crying girl face",
    "depressed", "depression", "depressed human", "depressed person", "depressed man", "depressed woman", "depressed boy", "depressed girl",
    "depressed portrait", "depressed human portrait", "depressed person portrait", "depressed man portrait", "depressed woman portrait", "depressed boy portrait", "depressed girl portrait",
    "lonely", "loneliness", "lonely human", "lonely person", "lonely man", "lonely woman", "lonely boy", "lonely girl",
    "upset", "upset human", "upset person", "upset man", "upset woman", "upset boy", "upset girl",
    "heartbroken", "heartbroken human", "heartbroken person", "heartbroken man", "heartbroken woman", "heartbroken boy", "heartbroken girl",
    "grief", "grieving", "grieving human", "grieving person", "grieving man", "grieving woman", "grieving boy", "grieving girl",
    "melancholic", "melancholic human", "melancholic person", "melancholic man", "melancholic woman", "melancholic boy", "melancholic girl",
    "unhappy", "unhappy human", "unhappy person", "unhappy man", "unhappy woman", "unhappy boy", "unhappy girl",
    "miserable", "miserable human", "miserable person", "miserable man", "miserable woman", "miserable boy", "miserable girl",
    "distressed", "distressed human", "distressed person", "distressed man", "distressed woman", "distressed boy", "distressed girl",
]

def get_images(query, page):
    params = {
        'key': API_KEY["pixabay"],
        'q': query,
        'image_type': 'photo',
        'per_page': str(200),
        'page': page,
    }
    path = folders_management("pixabay")

    try:
        url = "https://pixabay.com/api/" + '?' + urllib.parse.urlencode(params)
        response = requests.get(url)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        response_json = response.json()
        print("q = ", query)
        print("page = ", page)
        print("total = ", response_json["total"])
        for hit in response_json['hits']:
            image_url = hit.get('previewURL')
            if image_url:
                # Définir le chemin de sauvegarde de l'image
                image_id = hit['id']
                file_extension = os.path.splitext(image_url)[1]  # Obtenir l'extension du fichier
                save_path = os.path.join(path, f'image_{image_id}{file_extension}')

                # Télécharger l'image
                download_image(image_url, save_path)
            else:
                print(f'Pas d\'URL d\'image disponible pour l\'ID : {hit["id"]}\n')


def main():
    i = 0
    for query in queries:
        for page in range(1, 4):
            get_images(query, str(page))
            print()
            i += 1
            if i == 99:
                time.sleep(120)
                i = 0
    print("END OF SCRAP", end="\n\n")

if __name__ == "__main__":
    main()