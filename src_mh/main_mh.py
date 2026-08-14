from typing import Generic, TypeVar

import pandas as pd

from src_mh.carregador_dados.carregar_dados_xlsx import CarregarDadosXLSX
from src_mh.carregador_dados.icarregar_dados import ICarregarDados
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_mh.estrategia_modelo.regressao_linear import RegressaoLinearEstrategia
from src_mh.prepara_dados.iprepara_dados import IPrepararDados
from src_mh.prepara_dados.preparar_dados import PrepararDadosDataFame


T = TypeVar("T", bound=pd.DataFrame)
X = TypeVar("X", bound=pd.DataFrame)
Y = TypeVar("Y", bound=pd.Series)


class PipelineML(Generic[T, X, Y]):

    def __init__(
        self,
        carregar_dados: ICarregarDados[T],
        prepara_dados: IPrepararDados[T, X, Y],
        modelos: list[EstrategiaModelo]
    ):
        self.__carregar_dados = carregar_dados
        self.__prepara_dados = prepara_dados
        self.__modelos = modelos

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

        print("\n==========================================")
        print("    INICIANDO TREINAMENTO DE MODELOS      ")
        print("==========================================")

        for modelo in self.__modelos:
            print(f"\n--- Treinando Modelo: {modelo.nome} ---")
            modelo.treinar(x_train, y_train)

            metricas = modelo.obter_resultados(x_test, y_test)
            print("\nMétricas de Avaliação no Conjunto de Teste:")
            for m, valor in metricas.items():
                print(m, valor)




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

    modelo_regressao_linear = RegressaoLinearEstrategia()

    pml = PipelineML[
        pd.DataFrame,
        pd.DataFrame,
        pd.Series
    ](
        carregar_dados=carregar_dados,
        prepara_dados=prepara_dados,
        modelos=[modelo_regressao_linear]
    )

    pml.rodar_treinamento_simples()