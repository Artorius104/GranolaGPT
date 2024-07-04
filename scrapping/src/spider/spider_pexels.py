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
    "pexels": "dWoNsK02EseecQnf2CshjkcM7GIwD4WTHQtqij52vUziu3YtkK6TJTtX"
}

queries = [
    "sad person", "sad man", "sad woman", "sad child",
    "sad person portrait", "sad man portrait", "sad woman portrait", "sad child portrait",
    "crying person portrait", "crying man portrait", "crying woman portrait", "crying child portrait",
    "depressed person portrait", "depressed man portrait", "depressed woman portrait", "depressed child portrait",
    "lonely person", "lonely man", "lonely woman", "lonely child",
    "unhappy person", "unhappy man", "unhappy woman", "unhappy child",
]


def count_requests(count):
    if count == 190:
        time.sleep(3600)
        return True
    else:
        return False

def get_images(url):
    headers = {
        'Authorization': API_KEY["pexels"],
    }
    path = folders_management("pexels")

    try:
        response = requests.get(url, headers=headers)
        response.raise_for_status()
    except HTTPError as http_err:
        print(f"HTTP error occurred: {http_err}")
    except Exception as err:
        print(f"Other error occurred: {err}")
    else:
        response_json = response.json()
        print(url)
        print(response_json["total_results"])
        for photo in response_json['photos']:
            image_url = photo['src']['medium']
            if image_url:
                # Définir le chemin de sauvegarde de l'image
                image_id = photo['id']
                file_extension = ".jpeg"  # Obtenir l'extension du fichier
                save_path = os.path.join(path, f'image_{image_id}{file_extension}')

                # Télécharger l'image
                download_image(image_url, save_path)
            else:
                print(f'Pas d\'URL d\'image disponible pour l\'ID : {photo["id"]}\n')

        print()
        next_page = response_json.get('next_page', None)
        if next_page is None:
            return None
        return next_page


def main():
    count = 0

    for query in queries:
        params = {
            'query': query,
            'locale': 'fr-FR',
            'page': str(1),
            'per_page': str(80)
        }
        url = "https://api.pexels.com/v1/search" + '?' + urllib.parse.urlencode(params)
        next_url = get_images(url)
        count += 1

        while next_url is not None:
            next_url = get_images(next_url)
            count += 1
            if count_requests(count):
                count = 0
        count += 1
        if count_requests(count):
            count = 0

if __name__ == "__main__":
    main()