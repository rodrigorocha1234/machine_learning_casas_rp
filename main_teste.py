from time import sleep
from typing import TypeVar

from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from selenium.webdriver.chrome.service import Service
from selenium.webdriver.common.by import By
from selenium.webdriver.support import expected_conditions as EC
from selenium.webdriver.support.wait import WebDriverWait
from webdriver_manager.chrome import ChromeDriverManager


options = Options()
# options.add_argument("--headless=new")
servico = Service(ChromeDriverManager().install())
driver = webdriver.Chrome(service=servico, options=options)
driver.maximize_window()
driver.get("https://www.lagoimobiliaria.com.br/pesquisa-de-imoveis/?locacao_venda=V&id_cidade%5B%5D=90&id_tipo_imovel%5B%5D=5&id_tipo_imovel%5B%5D=19&finalidade=residencial&dormitorio=&garagem=&vmi=&vma=")

dados_imoveis = driver.find_elements(By.CLASS_NAME, "muda_card1")
print(len(dados_imoveis))




sleep(20)
for dado in dados_imoveis:
    try:
        print()
        print(dado.find_element(By.CLASS_NAME, "cod-imovel").text)
        print(dado.find_element(By.CLASS_NAME, "card-titulo").text)
        print(dado.find_element(By.CLASS_NAME, "card-valores").text.strip())
        print(dado.find_element(By.CLASS_NAME, "card-bairro-cidade-texto").text.strip())
        print(dado.find_element(By.CLASS_NAME, "dorm-ico").text.strip().split()[0])
        print(dado.find_element(By.CLASS_NAME, "banh-ico").text.strip().split()[0])
        print(dado.find_element(By.CLASS_NAME, "gar-ico").text.strip().split()[0])





    except:
        continue
    print()
# pagination = driver.find_element(By.CLASS_NAME, "pagination")
#
#
#
# driver.execute_script("""
#     arguments[0].scrollIntoView({
#         behavior: 'instant',
#         block: 'center'
#     });
# """, pagination)
# botao_proxima = WebDriverWait(driver, 10).until(
#     EC.element_to_be_clickable(
#         (By.CSS_SELECTOR, 'body > main > section > div > div.pagination > ul > li:nth-child(9) > a')
#     )
# )
# botao_proxima.click()

ultimo_link = WebDriverWait(driver, 10).until(
    EC.element_to_be_clickable(
        (By.XPATH, "//div[contains(@class,'pagination')]//li[last()]/a")
    )
)

driver.execute_script("arguments[0].scrollIntoView({block:'center'});", ultimo_link)
driver.execute_script("arguments[0].click();", ultimo_link)





sleep(80)
driver.quit()