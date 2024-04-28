# Pour setup Selenium avec chrome driver :
# https://github.com/password123456/setup-selenium-with-chrome-driver-on-ubuntu_debian
# import des packages pour le téléchargement
import requests
import os

# import des packages pour le scrapping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager


# ENV Variables : URLs des pages à scrap + mood
# Paths env variables
absolute_path = os.path.dirname(__file__)
res_path = os.path.join(absolute_path, "../../res")

# URLs env variables
happy_getty_page_url = "https://www.gettyimages.fr/photos/portrait-sourire?assettype=image&phrase=portrait%20sourire&sort=mostpopular&license=rf%2Crm"
neutral_getty_page_url = "https://www.gettyimages.fr/photos/portrait-sans-expression?assettype=image&phrase=portrait%20sans%20expression&sort=mostpopular&license=rf%2Crm"
sad_getty_page_url = "https://www.gettyimages.fr/photos/portrait-triste?assettype=image&phrase=portrait%20triste&sort=mostpopular&license=rf%2Crm"
happy_google_page_url = "https://www.google.com/search?q=photo+portrait+sourire&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0-czAMsrZBzi2iU5nsLoyAeW59zCg%3A1714189767680&ei=x3UsZr2SKZ-okdUP55CcoAo&ved=0ahUKEwj9kbjBvuGFAxUfVKQEHWcIB6QQ4dUDCBA&uact=5&oq=portrait+sourire&gs_lp=Egxnd3Mtd2l6LXNlcnAiEHBvcnRyYWl0IHNvdXJpcmUyBRAAGIAEMgUQABiABDIGEAAYCBgeSKA7ULoQWP85cAF4AJABAJgBNaABmgWqAQIxNrgBA8gBAPgBAZgCEKAC0AXCAgQQIxgnwgIKEAAYgAQYQxiKBcICCBAAGIAEGLEDwgINEAAYgAQYsQMYQxiKBcICBBAAGB7CAgYQABgFGB7CAgcQABiABBgYmAMAiAYBkgcCMTagB-lN&sclient=gws-wiz-serp"
neutral_google_page_url = "https://www.google.com/search?q=photo+portrait+sans+expression&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0-q0CU5qbyrk0IL2F0vWzSLXCAVZQ%3A1714189777003&ei=0HUsZrnvPIKfkdUPzqSAOA&ved=0ahUKEwi5l_HFvuGFAxWCT6QEHU4SAAcQ4dUDCBA&uact=5&oq=portrait+sans+expression&gs_lp=Egxnd3Mtd2l6LXNlcnAiGHBvcnRyYWl0IHNhbnMgZXhwcmVzc2lvbkj6swJQ9oICWNqxAnALeACQAQCYAUigAccHqgECMjO4AQPIAQD4AQGYAhigAuQEwgIKEAAYgAQYQxiKBcICBRAAGIAEwgIGEAAYCBgewgIEEAAYHsICBBAjGCfCAggQABiABBixA8ICBBAAGAPCAgYQABgFGB6YAwCIBgGSBwIyNKAHnkg&sclient=gws-wiz-serp"
sad_google_page_url = "https://www.google.com/search?q=photo+portrait+triste+couleur&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0_arSQYgVtLtugkf5BZqWSoK4965A%3A1714189873147&ei=MXYsZrrQCJP2kdUPmYmx-AM&ved=0ahUKEwj6rN3zvuGFAxUTe6QEHZlEDD8Q4dUDCBA&uact=5&oq=portrait+triste+couleur&gs_lp=Egxnd3Mtd2l6LXNlcnAiF3BvcnRyYWl0IHRyaXN0ZSBjb3VsZXVySPIVUKsDWMMUcAF4AJABAJgBK6ABvQKqAQE4uAEDyAEA-AEBmAIDoAJWwgIKEAAYgAQYQxiKBcICBRAAGIAEwgIGEAAYCBgewgIEEAAYHpgDAIgGAZIHATOgB88I&sclient=gws-wiz-serp"


# SCRAPPING
def scrapping_images(driver, source: str, page_url: str) -> list[dict[int, str]]:
    i = 0
    element_list = []

    for page in range(1, 2, 1):
        url = page_url + "&page=" + str(page)
        driver.get(url)
        if source == "getty":
            img_elements = driver.find_elements(By.TAG_NAME, "source")
            for img_element in img_elements:
                img_url = img_element.get_attribute("srcset")
                element_list.append({"id": i, "url": img_url})
                i += 1
        else:
            img_elements = driver.find_elements(By.TAG_NAME, "img")
            for img_element in img_elements:
                img_url = img_element.get_attribute("src")
                element_list.append({"id": i, "url": img_url})
                i += 1

    return element_list


# TELECHARGEMENT
def download_images(element_list: list[dict[int, str]], path: str) -> None:
    for index, img_info in enumerate(element_list):
        img_url = img_info['url']
        response = requests.get(img_url)

        if response.status_code == 200:
            with open(f'{path}/scrapped_{index}.jpg', 'wb') as f:
                f.write(response.content)
                print(f"Image {index} téléchargée avec succès.")
        else:
            print(f"Impossible de télécharger l'image {index}. Statut de la requête : {response.status_code}")


def main():
    # Mettre la catégorie des images à scrapper : happy, sad, neutral
    mood = "happy"
    path = res_path + '/' + mood
    # Création du répertoire happy_images si non existant
    if not os.path.exists(path):
        os.makedirs(path)

    # Série d'option nécessaire au fonctionnement de Selenium (à voir pourquoi)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    # Liste de dictionnaires pour stocker les URLs
    element_list = scrapping_images(driver, "getty", happy_getty_page_url)
    print(element_list)
    download_images(element_list, path)

    # Fermeture du driver
    driver.close()

if __name__ == "__main__":
    main()