from typing import Generic, TypeVar

import pandas as pd

from src_mh.carregador_dados.carregar_dados_xlsx import CarregarDadosXLSX
from src_mh.carregador_dados.icarregar_dados import ICarregarDados
from src_mh.prepara_dados.iprepara_dados import IPrepararDados
from src_mh.prepara_dados.preparar_dados import PrepararDadosDataFame

T = TypeVar("T", bound=pd.DataFrame)
U = TypeVar("U", bound=pd.DataFrame)


class PipelineML(Generic[T, U]):
    def __init__(self, carregar_dados: ICarregarDados[T], prepara_dados: IPrepararDados[U]):
        self.__carregar_dados = carregar_dados
        self.__prepara_dados = prepara_dados

    def rodar_preparacao_dados(self) -> tuple[T, T, T, T]:
        base: T = self.__carregar_dados.carregar_dados()
        base: T = self.__prepara_dados.realizar_engenharia_atributos(base)
        x_train, x_test, y_train, y_test = self.__prepara_dados.separar_treino_teste(base)
        return x_train, x_test, y_train, y_test

    def rodar_treinamento_simples(self):
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()


if __name__ == "__main__":
    pml = PipelineML[pd.DataFrame, pd.DataFrame](
        carregar_dados=CarregarDadosXLSX(
            colunas=['Metragem', 'Quartos', 'Banheiros', 'Vagas', 'Bairro',
                     'Valor_da_Venda']),
        prepara_dados=PrepararDadosDataFame()

    )
    pml.rodar_treinamento_simples()
