import logging
from typing import Any, Generic, Sequence, TypeVar

import pandas as pd

from src_mh.carregador_dados.carregar_dados_xlsx import CarregarDadosXLSX
from src_mh.carregador_dados.icarregar_dados import ICarregarDados
from src_mh.config import obter_config_carregador_dados, obter_config_mlflow
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_mh.estrategia_modelo.regressao_linear import RegressaoLinearEstrategia
from src_mh.observadores.console_observador import ConsoleObservador
from src_mh.observadores.iobservador_ml import IObservadorML
from src_mh.observadores.mlflow_observador import MLflowObservador
from src_mh.prepara_dados.iprepara_dados import IPrepararDados
from src_mh.prepara_dados.preparar_dados import PrepararDadosDataFame

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

T = TypeVar("T", bound=pd.DataFrame)
X = TypeVar("X", bound=pd.DataFrame)
Y = TypeVar("Y", bound=pd.Series)


class PipelineML(Generic[T, X, Y]):

    def __init__(
            self,
            carregar_dados: ICarregarDados[T],
            prepara_dados: IPrepararDados[T, X, Y],
            modelos: Sequence[EstrategiaModelo[X, Y, Any]],
            observadores: list[IObservadorML] | None = None,
    ):
        self.__carregar_dados = carregar_dados
        self.__prepara_dados = prepara_dados
        self.__modelos = modelos
        self.__observadores: list[IObservadorML] = observadores or []

    def adicionar_observador(self, observador: IObservadorML) -> None:
        """Adiciona um novo observador ao pipeline (GoF Observer Pattern)."""
        self.__observadores.append(observador)

    def __notificar_inicio_experimento(
            self, nome_experimento: str, parametros: dict[str, object]
    ) -> None:
        for obs in self.__observadores:
            obs.iniciar_experimento(nome_experimento, parametros)

    def __notificar_metricas(
            self, nome_modelo: str, metricas: dict[str, object]
    ) -> None:
        for obs in self.__observadores:
            obs.registrar_metricas(nome_modelo, metricas)

    def __notificar_fim_experimento(self) -> None:
        for obs in self.__observadores:
            obs.finalizar_experimento()

    def rodar_preparacao_dados(self) -> tuple[X, X, Y, Y]:
        # 1. Carregamento
        base: T = self.__carregar_dados.carregar_dados()

        # 2. Engenharia de atributos
        base_dois: T = self.__prepara_dados.realizar_engenharia_atributos(base)

        x_train, x_test, y_train, y_test = self.__prepara_dados.separar_treino_teste(
            base_dois, tipo_escalonamento=None
        )

        return x_train, x_test, y_train, y_test

    def rodar_treinamento_simples(self) -> None:
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()

        logger.info("INICIANDO TREINAMENTO DE MODELOS NO PIPELINE ML")

        # Notifica início do experimento para todos os observadores registrados
        self.__notificar_inicio_experimento(
            "Treinamento_Imoveis_RP",
            {
                "n_amostras_treino": len(x_train),
                "n_amostras_teste": len(x_test),
                "n_modelos": len(self.__modelos),
            },
        )

        for modelo in self.__modelos:
            logger.info("Treinando Modelo: %s", modelo.nome)
            modelo.treinar(x_train, y_train)

            # Obtenção individual e explícita por cada método dedicado do modelo
            metricas = modelo.obter_resultados(x_test, y_test)
            equacao_geral = modelo.obter_equacao_reta_geral()
            interceptos_zona = modelo.obter_equacoes_por_zona()
            figura_underfit = modelo.gerar_figura_underfit_overfit(x_test, y_test)
            resultado_cv = modelo.realizar_validacao_cruzada(x_train, y_train, iteracao=42)

            resultados_completos: dict[str, object] = {
                **metricas,
            }

            if equacao_geral:
                resultados_completos["equacao_reta_geral"] = equacao_geral

            if interceptos_zona:
                resultados_completos["interceptos_por_zona"] = interceptos_zona

            if figura_underfit is not None:
                resultados_completos["figura_underfit_overfit"] = figura_underfit

            if resultado_cv:
                resultados_completos["validacao_cruzada"] = resultado_cv

            if modelo.modelo_objeto is not None:
                resultados_completos["modelo_objeto"] = modelo.modelo_objeto
                x_sample = x_test.head(5)
                resultados_completos["x_sample"] = x_sample
                resultados_completos["y_sample"] = modelo.predizer(x_sample)

            logger.info("Resultados de %s: %s", modelo.nome, resultados_completos)

            # Notifica os observadores transmitindo o dicionário unificado
            self.__notificar_metricas(modelo.nome, resultados_completos)

        # Notifica conclusão do experimento
        self.__notificar_fim_experimento()

    def realizar_tuning_parametros(self) -> None:
        """Alias amigável para realizar_turing_parametros."""
        """Executa a busca em grade (GridSearchCV / Tuning) para todos os modelos e envia os resultados aos observadores (MLflow, Console)."""
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()
        x_completo = pd.concat([x_train, x_test], axis=0)
        y_completo = pd.concat([y_train, y_test], axis=0)

        logger.info("INICIANDO BUSCA EM GRADE (GRIDSEARCHCV / TUNING) DE MODELOS NO PIPELINE ML")

        self.__notificar_inicio_experimento(
            "Tuning_Parametros_Imoveis_RP",
            {
                "n_amostras_totais": len(x_completo),
                "n_modelos": len(self.__modelos),
            },
        )

        for modelo in self.__modelos:
            logger.info("Executando Grid Search para Modelo: %s", modelo.nome)
            grid_search = modelo.realizar_grid_search(x_completo, y_completo)
            resultados_grid = modelo.obter_resultado_grid_search(grid_search)

            resultados_completos: dict[str, object] = {
                **resultados_grid,
                "melhores_parametros": grid_search.best_params_,
                "melhor_score": float(grid_search.best_score_),
            }

            if hasattr(grid_search, "best_estimator_") and grid_search.best_estimator_ is not None:
                resultados_completos["modelo_objeto"] = grid_search.best_estimator_
                x_sample = x_test.head(5)
                resultados_completos["x_sample"] = x_sample
                try:
                    resultados_completos["y_sample"] = grid_search.best_estimator_.predict(x_sample)
                except Exception as e:
                    logger.warning("Falha ao calcular y_sample para a assinatura no MLflow: %s", e)

            logger.info("Resultados de Grid Search de %s: %s", modelo.nome, resultados_completos)

            # Notifica os observadores transmitindo o dicionário de resultados do Tuning
            self.__notificar_metricas(modelo.nome, resultados_completos)

        # Notifica conclusão do experimento
        self.__notificar_fim_experimento()


if __name__ == "__main__":
    cfg_dados = obter_config_carregador_dados()
    cfg_mlflow = obter_config_mlflow()

    carregar_dados = CarregarDadosXLSX(
        colunas=cfg_dados.get(
            "colunas",
            [
                "Metragem",
                "Quartos",
                "Banheiros",
                "Vagas",
                "Bairro",
                "Valor_da_Venda",
            ],
        )
    )

    prepara_dados: IPrepararDados[
        pd.DataFrame, pd.DataFrame, pd.Series
    ] = PrepararDadosDataFame()

    # Instancia os modelos lendo as configurações centralizadas de model_config.yaml
    modelo_regressao_linear = RegressaoLinearEstrategia()

    # Instanciando observadores (MLflow + Console) com suporte ao Model Registry centralizado
    mlflow_obs = MLflowObservador(
        tracking_uri=cfg_mlflow.get("tracking_uri", "http://localhost:5000"),
        nome_experimento=cfg_mlflow.get("nome_experimento", "Experimento_Previsão_Apartamentos_RP"),
        nome_modelo_registry=cfg_mlflow.get("nome_modelo_registry", "Modelo_Preco_Imoveis_RP"),
    )
    console_obs = ConsoleObservador()

    pml = PipelineML[pd.DataFrame, pd.DataFrame, pd.Series](
        carregar_dados=carregar_dados,
        prepara_dados=prepara_dados,
        modelos=[modelo_regressao_linear],
        observadores=[console_obs, mlflow_obs],  # Registro dos observadores
    )
    #
    # pml.rodar_treinamento_simples()
    pml.realizar_turing_parametros()
