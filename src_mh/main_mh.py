from typing import Generic, TypeVar

import pandas as pd

from src_mh.carregador_dados.carregar_dados_xlsx import CarregarDadosXLSX
from src_mh.carregador_dados.icarregar_dados import ICarregarDados
from src_mh.prepara_dados.iprepara_dados import IPrepararDados
from src_mh.prepara_dados.preparar_dados import PrepararDadosDataFame


T = TypeVar("T", bound=pd.DataFrame)
X = TypeVar("X", bound=pd.DataFrame)
Y = TypeVar("Y")


class PipelineML(Generic[T, X, Y]):

    def __init__(
        self,
        carregar_dados: ICarregarDados[T],
        prepara_dados: IPrepararDados[T, X, Y]
    ):
        self.__carregar_dados = carregar_dados
        self.__prepara_dados = prepara_dados

    def rodar_preparacao_dados(
        self
    ) -> tuple[X, X, Y, Y]:

        # 1. Carregamento
        base: T = self.__carregar_dados.carregar_dados()

        # 2. Engenharia de atributos
        base_dois: T = (
            self.__prepara_dados
            .realizar_engenharia_atributos(base)
        )

        print(base_dois)

        # 3. Train/Test Split + preprocessing
        x_train, x_test, y_train, y_test = (
            self.__prepara_dados
            .separar_treino_teste(
                base_dois,
                tipo_escalonamento=None
            )
        )

        return x_train, x_test, y_train, y_test

    def rodar_treinamento_simples(self) -> None:

        x_train, x_test, y_train, y_test = (
            self.rodar_preparacao_dados()
        )

        print(x_train)


if __name__ == "__main__":

    carregar_dados = CarregarDadosXLSX(
        colunas=[
            "Metragem",
            "Quartos",
            "Banheiros",
            "Vagas",
            "Bairro",
            "Valor_da_Venda"
        ]
    )

    prepara_dados: IPrepararDados[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series
    ] = PrepararDadosDataFame()

    pml = PipelineML[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series
    ](
        carregar_dados=carregar_dados,
        prepara_dados=prepara_dados
    )

    pml.rodar_treinamento_simples()