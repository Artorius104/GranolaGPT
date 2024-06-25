import os

# import des packages pour le scrapping
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from webdriver_manager.chrome import ChromeDriverManager

from utils import folders_management, download_image


# Paths env variables
absolute_path = os.path.dirname(__file__)
res_path = os.path.join(absolute_path, "../../../dataset")

data_sources = [
    # {
    #     "website": "google",
    #     "url": "https://www.google.com/search?q=photo+portrait+sourire&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0-czAMsrZBzi2iU5nsLoyAeW59zCg%3A1714189767680&ei=x3UsZr2SKZ-okdUP55CcoAo&ved=0ahUKEwj9kbjBvuGFAxUfVKQEHWcIB6QQ4dUDCBA&uact=5&oq=portrait+sourire&gs_lp=Egxnd3Mtd2l6LXNlcnAiEHBvcnRyYWl0IHNvdXJpcmUyBRAAGIAEMgUQABiABDIGEAAYCBgeSKA7ULoQWP85cAF4AJABAJgBNaABmgWqAQIxNrgBA8gBAPgBAZgCEKAC0AXCAgQQIxgnwgIKEAAYgAQYQxiKBcICCBAAGIAEGLEDwgINEAAYgAQYsQMYQxiKBcICBBAAGB7CAgYQABgFGB7CAgcQABiABBgYmAMAiAYBkgcCMTagB-lN&sclient=gws-wiz-serp",
    #     "mood": "happy"
    # },
    # {
    #     "website": "google",
    #     "url": "https://www.google.com/search?q=photo+portrait+sans+expression&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0-q0CU5qbyrk0IL2F0vWzSLXCAVZQ%3A1714189777003&ei=0HUsZrnvPIKfkdUPzqSAOA&ved=0ahUKEwi5l_HFvuGFAxWCT6QEHU4SAAcQ4dUDCBA&uact=5&oq=portrait+sans+expression&gs_lp=Egxnd3Mtd2l6LXNlcnAiGHBvcnRyYWl0IHNhbnMgZXhwcmVzc2lvbkj6swJQ9oICWNqxAnALeACQAQCYAUigAccHqgECMjO4AQPIAQD4AQGYAhigAuQEwgIKEAAYgAQYQxiKBcICBRAAGIAEwgIGEAAYCBgewgIEEAAYHsICBBAjGCfCAggQABiABBixA8ICBBAAGAPCAgYQABgFGB6YAwCIBgGSBwIyNKAHnkg&sclient=gws-wiz-serp",
    #     "mood": "neutral"
    # },
    # {
    #     "website": "google",
    #     "url": "https://www.google.com/search?q=photo+portrait+triste+couleur&sca_esv=4ec8c5ee11c012e4&rlz=1C1CHBF_frFR1089FR1089&udm=2&biw=958&bih=1078&sxsrf=ACQVn0_arSQYgVtLtugkf5BZqWSoK4965A%3A1714189873147&ei=MXYsZrrQCJP2kdUPmYmx-AM&ved=0ahUKEwj6rN3zvuGFAxUTe6QEHZlEDD8Q4dUDCBA&uact=5&oq=portrait+triste+couleur&gs_lp=Egxnd3Mtd2l6LXNlcnAiF3BvcnRyYWl0IHRyaXN0ZSBjb3VsZXVySPIVUKsDWMMUcAF4AJABAJgBK6ABvQKqAQE4uAEDyAEA-AEBmAIDoAJWwgIKEAAYgAQYQxiKBcICBRAAGIAEwgIGEAAYCBgewgIEEAAYHpgDAIgGAZIHATOgB88I&sclient=gws-wiz-serp",
    #     "mood": "sad"
    # }
    {
        "website": "pixabay",
        "url": "https://pixabay.com/fr/images/search/portrait%20triste%20humain/",
        "mood": "sad"
    }
]


def scrapping_images(driver, website: str, page_url: str) -> list[dict[int, str]]:
    i = 0
    element_list = []

    for page in range(1, 2, 1):
        url = page_url + "&page=" + str(page)
        driver.get(url)
        if website == "getty":
            img_elements = driver.find_elements(By.TAG_NAME, "source")
            for img_element in img_elements:
                img_url = img_element.get_attribute("srcset")
                element_list.append({"id": i, "url": img_url})
                i += 1
        else:
            img_elements = driver.find_elements(By.CLASS_NAME, "YQ4gaf")
            for img_element in img_elements:
                img_url = img_element.get_attribute("src")
                element_list.append({"id": i, "url": img_url})
                i += 1

    return element_list


def main():
    # Série d'option nécessaire au fonctionnement de Selenium (à voir pourquoi)
    options = Options()
    options.add_argument('--headless')
    options.add_argument('--no-sandbox')
    options.add_argument('--disable-dev-shm-usage')
    driver = webdriver.Chrome(service=Service(ChromeDriverManager().install()), options=options)

    for source in data_sources:
        # Création du répertoire happy_images si non existant
        path = folders_management(source)
        # Liste de dictionnaires pour stocker les URLs
        element_list = scrapping_images(driver, source["website"], source["url"])
        # print(element_list)
        download_images(element_list, source["website"], path)

    # Fermeture du driver
    driver.close()

if __name__ == "__main__":
    main()