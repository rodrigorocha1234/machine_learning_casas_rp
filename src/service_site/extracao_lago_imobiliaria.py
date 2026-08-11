import re
from time import sleep
from typing import Generator

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


class ExtracaoLagoImobiliaria:
    def __init__(self, url: str):
        self.__options = Options()
        self.__options.add_argument("--headless=new")
        self.__servico = Service(ChromeDriverManager().install())
        self.__driver = webdriver.Chrome(service=self.__servico, options=self.__options)
        self.__driver.maximize_window()
        self.__url = url

    def abrir_site(self):
        self.__driver.get(self.__url)

    def coletar_dados(self) -> Generator[dict[str, str | int], None, None]:
        dados_imoveis = self.__driver.find_elements(By.CLASS_NAME, "muda_card1")

        for dado in dados_imoveis:
            try:
                yield {'codigo': dado.find_element(By.CLASS_NAME, "cod-imovel").text,
                       'apartamento': dado.find_element(By.CLASS_NAME, "card-titulo").text,
                       'valor': dado.find_element(By.CLASS_NAME, "card-valores").text.strip(),
                       'bairro': dado.find_element(By.CLASS_NAME, "card-bairro-cidade-texto").text.strip(),
                       'qtd_quartos': dado.find_element(By.CLASS_NAME, "dorm-ico").text.strip().split()[0],
                       'qtd_banheiros': dado.find_element(By.CLASS_NAME, "banh-ico").text.strip().split()[0],
                       'qtd_graragem': dado.find_element(By.CLASS_NAME, "gar-ico").text.strip().split()[0],
                       'link': dado.find_element(By.CSS_SELECTOR, "a.carousel-cell").get_attribute("href") or ""}
            except:
                continue

    def executar_paginacao(self) -> bool:
        try:
            ultimo_link = WebDriverWait(self.__driver, 10).until(
                EC.element_to_be_clickable((By.XPATH, "//div[contains(@class,'pagination')]//li[last()]/a")))

            self.__driver.execute_script("arguments[0].scrollIntoView({block:'center'});", ultimo_link)
            self.__driver.execute_script("arguments[0].click();", ultimo_link)
            return True
        except:
            return False

    def obter_metragem(self):
        sleep(10)
        self.__driver.execute_script(
            "window.scrollTo(0, document.body.scrollHeight);"
        )
        area = self.__driver.find_element(By.CSS_SELECTOR, "strong.fw-bold").text.replace("m²", "").strip()

        match = re.search(r"\d+(?:\.\d+)?", area)

        if match:
            numero = float(match.group())
            print(numero)

            return int(numero)

    def fechar_site(self):
        self.__driver.quit()
