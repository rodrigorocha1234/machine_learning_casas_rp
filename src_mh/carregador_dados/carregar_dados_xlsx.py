import os

import pandas as pd

from src_mh.carregador_dados.icarregar_dados import ICarregarDados

pd.set_option("display.max_columns", None)  # Exibe todas as colunas
pd.set_option("display.max_rows", 50)  # Máximo de linhas exibidas
pd.set_option("display.width", 200)  # Largura do DataFrame
pd.set_option("display.max_colwidth", 50)  # Largura máxima das colunas

pd.set_option("display.expand_frame_repr", False)  # Evita quebrar o DataFrame


class CarregarDadosXLSX:

    def __init__(self, colunas: list[str]):
        self.__colunas = colunas
        self.__caminho_arquivo = os.path.join(os.getcwd(), "dados_imoveis/bairro_final_v3_engineered.xlsx")

    def carregar_dados(self) -> pd.DataFrame:
        base = pd.read_excel(self.__caminho_arquivo, usecols=self.__colunas)
        return base
