import logging
from typing import Any, Generic, Sequence, TypeVar

import numpy as np
import pandas as pd

from src_mh.carregador_dados.carregar_dados_xlsx import CarregarDadosXLSX
from src_mh.carregador_dados.icarregar_dados import ICarregarDados
from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo
from src_mh.estrategia_modelo.regressao_linear import RegressaoLinearEstrategia
from src_mh.estrategia_modelo.regressao_polinomial import RegressaoPolinomialEstrategia
from src_mh.estrategia_modelo.regressao_ridge import RegressaoRidgeEstrategia
from src_mh.estrategia_modelo.regressao_lasso import RegressaoLassoEstrategia
from src_mh.estrategia_modelo.regressao_elasticnet import RegressaoElasticNetEstrategia
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
            self,  metricas: dict[str, object]
    ) -> None:
        for obs in self.__observadores:
            obs.registrar_metricas(metricas)

    def __notificar_fim_experimento(self) -> None:
        for obs in self.__observadores:
            obs.finalizar_experimento()



    def rodar_preparacao_dados(self) -> tuple[X, X, Y, Y]:
        base: T = self.__carregar_dados.carregar_dados()
        base_dois: T = self.__prepara_dados.realizar_engenharia_atributos(base)
        x_train, x_test, y_train, y_test = self.__prepara_dados.separar_treino_teste(
            base_dois, tipo_escalonamento=None
        )
        return x_train, x_test, y_train, y_test

    def rodar_treinamento_simples(self) -> dict[str, dict[str, object]]:
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()

        logger.info("INICIANDO TREINAMENTO DE MODELOS NO PIPELINE ML")

        # Notifica início do experimento pai para todos os observadores registrados
        self.__notificar_inicio_experimento(
            "Treinamento_Imoveis_RP",
            {
                "n_amostras_treino": len(x_train),
                "n_amostras_teste": len(x_test),
                "n_modelos": len(self.__modelos),
            },
        )


        resultados_por_modelo: dict[str, dict[str, object]] = {}

        for modelo in self.__modelos:
            logger.info("Treinando Modelo: %s", modelo.nome)

            # Inicia uma Run dedicada por modelo de regressão no MLflow
            self.__notificar_inicio_experimento(
                f"{modelo.nome}_Treinamento_Simples",
                {
                    "modelo": modelo.nome,
                    "n_amostras_treino": len(x_train),
                    "n_amostras_teste": len(x_test),
                },
            )

            modelo.treinar(x_train, y_train)

            # Obtenção individual e explícita por cada método dedicado do modelo
            metricas = modelo.obter_resultados(x_test, y_test)
            equacao_geral = modelo.obter_equacao_reta_geral()
            interceptos_zona = modelo.obter_equacoes_por_zona()
            dados_curva = modelo.obter_curva_validacao(x_test, y_test)
            figura_underfit = modelo.gerar_figura_underfit_overfit(dados_curva)

            resultados_completos: dict[str, object] = {
                **metricas,
                "nome_modelo": modelo.nome,
                "equacao_geral": equacao_geral,
                "figura_underfit": figura_underfit,
                **dados_curva,
                **interceptos_zona,
            }

            if modelo.modelo_objeto is not None:
                resultados_completos["modelo_objeto"] = modelo.modelo_objeto
                x_sample = x_test.head(5)
                resultados_completos["x_sample"] = x_sample
                resultados_completos["y_sample"] = modelo.predizer(x_sample)

            resultados_por_modelo[modelo.nome] = resultados_completos

            # Notifica os observadores transmitindo o dicionário unificado por modelo
            self.__notificar_metricas(resultados_completos)

            # Finaliza a run dedicada do modelo
            self.__notificar_fim_experimento()

        # Loga as métricas resumidas de cada modelo na Run Pai para aparecer na tela principal do MLflow
        metricas_resumo_pai: dict[str, object] = {
            "nome_modelo": "Resumo_Comparativo_Modelos"
        }
        for nome_mod, res in resultados_por_modelo.items():
            for metric_key in ("r2", "rmse", "mae", "medae", "smape"):
                if metric_key in res:
                    metricas_resumo_pai[f"{nome_mod}_{metric_key}"] = res[metric_key]

        self.__notificar_metricas(metricas_resumo_pai)

        # Finaliza a Run Pai do experimento
        self.__notificar_fim_experimento()

        return resultados_por_modelo



    def realizar_tuning_parametros(self) -> None:
        """Executa a busca em grade de hiperparâmetros (GridSearchCV) e notifica os observadores (MLflow, etc.)."""
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()

        logger.info("INICIANDO TUNING DE HIPERPARÂMETROS NO PIPELINE ML")

        self.__notificar_inicio_experimento(
            "Tuning_Hiperparametros_Imoveis_RP",
            {
                "n_amostras_totais": len(x_train) + len(x_test),
                "n_modelos": len(self.__modelos),
            },
        )

        for modelo in self.__modelos:
            logger.info("Executando Grid Search para o modelo: %s", modelo.nome)

            self.__notificar_inicio_experimento(
                f"{modelo.nome}_Tuning_Hiperparametros",
                {
                    "modelo": modelo.nome,
                    "n_amostras_totais": len(x_train) + len(x_test),
                },
            )

            grid_search = modelo.realizar_grid_search(x_train, y_train)

            resultados_completos = modelo.obter_resultado_grid_search(grid_search)
            resultados_completos["nome_modelo"] = modelo.nome

            if hasattr(grid_search, "best_estimator_") and grid_search.best_estimator_ is not None:
                resultados_completos["modelo_objeto"] = grid_search.best_estimator_
                x_sample = x_test.head(5)
                resultados_completos["x_sample"] = x_sample
                resultados_completos["y_sample"] = grid_search.best_estimator_.predict(x_sample)

            logger.info(
                "Resultados do Tuning de %s: %s", modelo.nome, resultados_completos
            )

            self.__notificar_metricas(resultados_completos)
            self.__notificar_fim_experimento()

        self.__notificar_fim_experimento()


    def realizar_validacao_cruzada(self, num_iteracoes: int = 30) -> None:
        """Executa a Validação Cruzada repetida de 30 iterações e grava no MLflow."""
        x_train, x_test, y_train, y_test = self.rodar_preparacao_dados()

        x_completo = pd.concat([x_train, x_test], axis=0).reset_index(drop=True)
        y_completo = pd.concat([y_train, y_test], axis=0).reset_index(drop=True)

        logger.info(
            "INICIANDO VALIDAÇÃO CRUZADA REPETIDA (%d ITERAÇÕES) NO PIPELINE ML",
            num_iteracoes,
        )

        self.__notificar_inicio_experimento(
            f"Validacao_Cruzada_Repetida_{num_iteracoes}_Runs",
            {
                "n_amostras_totais": len(x_completo),
                "num_iteracoes": num_iteracoes,
                "n_modelos": len(self.__modelos),
            },
        )

        for modelo in self.__modelos:
            logger.info(
                "Executando Validação Cruzada para %s (%d repetições)",
                modelo.nome,
                num_iteracoes,
            )

            rmse_scores: list[float] = []
            r2_scores: list[float] = []
            mae_scores: list[float] = []
            oof_rmse_scores: list[float] = []
            oof_r2_scores: list[float] = []
            oof_mae_scores: list[float] = []

            for i in range(num_iteracoes):
                self.__notificar_inicio_experimento(
                    f"{modelo.nome}_Iteracao_{i + 1}",
                    {
                        "modelo": modelo.nome,
                        "iteracao": i + 1,
                        "random_state_kfold": i,
                    },
                )

                resultado_cv = modelo.realizar_validacao_cruzada(
                    x_completo, y_completo, i
                )

                mean_scores = resultado_cv.get("mean_scores", {})
                if "mean_test_rmse" in mean_scores:
                    rmse_scores.append(mean_scores["mean_test_rmse"])
                if "mean_test_r2" in mean_scores:
                    r2_scores.append(mean_scores["mean_test_r2"])
                if "mean_test_mae" in mean_scores:
                    mae_scores.append(mean_scores["mean_test_mae"])
                if "oof_completo_rmse" in mean_scores:
                    oof_rmse_scores.append(mean_scores["oof_completo_rmse"])
                if "oof_completo_r2" in mean_scores:
                    oof_r2_scores.append(mean_scores["oof_completo_r2"])
                if "oof_completo_mae" in mean_scores:
                    oof_mae_scores.append(mean_scores["oof_completo_mae"])

                resultado_cv["nome_modelo"] = modelo.nome
                resultado_cv.update(mean_scores)
                self.__notificar_metricas(resultado_cv)
                self.__notificar_fim_experimento()

            metricas_globais: dict[str, object] = {
                "nome_modelo": f"{modelo.nome}_Resumo_Global",
                "cv_30_runs_mean_r2": round(float(np.mean(r2_scores)), 4) if r2_scores else 0.0,
                "cv_30_runs_std_r2": round(float(np.std(r2_scores)), 4) if r2_scores else 0.0,
                "cv_30_runs_mean_rmse": round(float(np.mean(rmse_scores)), 2) if rmse_scores else 0.0,
                "cv_30_runs_std_rmse": round(float(np.std(rmse_scores)), 2) if rmse_scores else 0.0,
                "cv_30_runs_mean_mae": round(float(np.mean(mae_scores)), 2) if mae_scores else 0.0,
                "cv_30_runs_std_mae": round(float(np.std(mae_scores)), 2) if mae_scores else 0.0,
                "oof_completo_30_runs_mean_r2": round(float(np.mean(oof_r2_scores)), 4) if oof_r2_scores else 0.0,
                "oof_completo_30_runs_std_r2": round(float(np.std(oof_r2_scores)), 4) if oof_r2_scores else 0.0,
                "oof_completo_30_runs_mean_rmse": round(float(np.mean(oof_rmse_scores)), 2) if oof_rmse_scores else 0.0,
                "oof_completo_30_runs_std_rmse": round(float(np.std(oof_rmse_scores)), 2) if oof_rmse_scores else 0.0,
                "oof_completo_30_runs_mean_mae": round(float(np.mean(oof_mae_scores)), 2) if oof_mae_scores else 0.0,
                "oof_completo_30_runs_std_mae": round(float(np.std(oof_mae_scores)), 2) if oof_mae_scores else 0.0,
                "scores_por_iteracao_rmse": rmse_scores,
                "scores_por_iteracao_r2": r2_scores,
                "scores_por_iteracao_mae": mae_scores,
                "oof_scores_por_iteracao_rmse": oof_rmse_scores,
                "oof_scores_por_iteracao_r2": oof_r2_scores,
                "oof_scores_por_iteracao_mae": oof_mae_scores,
            }

            if modelo.modelo_objeto is not None:
                x_sample = x_test.head(5)
                try:
                    y_sample = modelo.predizer(x_sample)
                except Exception:
                    modelo.treinar(x_completo, y_completo)
                    y_sample = modelo.predizer(x_sample)

                metricas_globais["modelo_objeto"] = modelo.modelo_objeto
                metricas_globais["x_sample"] = x_sample
                metricas_globais["y_sample"] = y_sample

            self.__notificar_inicio_experimento(
                f"{modelo.nome}_Resumo_Global_{num_iteracoes}_Runs",
                {"num_iteracoes": num_iteracoes},
            )
            self.__notificar_metricas(metricas_globais)
            self.__notificar_fim_experimento()

        self.__notificar_fim_experimento()


if __name__ == "__main__":
    carregar_dados = CarregarDadosXLSX(
        colunas=[
            "Metragem",
            "Quartos",
            "Banheiros",
            "Vagas",
            "Bairro",
            "Valor_da_Venda",
        ]
    )

    prepara_dados: IPrepararDados[
        pd.DataFrame, pd.DataFrame, pd.Series
    ] = PrepararDadosDataFame()

    modelo_regressao_linear = RegressaoLinearEstrategia(params={
        'fit_intercept': Config.fit_intercept_rl
    })
    modelo_regressao_ridge = RegressaoRidgeEstrategia(params={
        'alpha': Config.alpha_ridge,
        'fit_intercept': Config.fit_intercept_ridge,
        'solver': Config.solver_ridge,
    })
    modelo_regressao_lasso = RegressaoLassoEstrategia(params={
        'alpha': Config.alpha_lasso,
        'fit_intercept': Config.fit_intercept_lasso,
        'max_iter': Config.max_iter_lasso,
    })
    modelo_regressao_elasticnet = RegressaoElasticNetEstrategia(params={
        'alpha': Config.alpha_elasticnet,
        'l1_ratio': Config.l1_ratio_elasticnet,
        'fit_intercept': Config.fit_intercept_elasticnet,
        'max_iter': Config.max_iter_elasticnet,
    })
    modelo_regressao_polinomial = RegressaoPolinomialEstrategia(params={
        'degree': Config.degree_rp,
        'include_bias': Config.include_bias_rp,
        'fit_intercept': Config.fit_intercept_rp,
        'positive': Config.positive_rp,
    })

    mlflow_obs = MLflowObservador(
        tracking_uri=Config.tracking_uri,
        nome_experimento=Config.nome_experimento,
        nome_modelo_registry=Config.nome_modelo_registry,
    )
    console_obs = ConsoleObservador()

    pml = PipelineML[pd.DataFrame, pd.DataFrame, pd.Series](
        carregar_dados=carregar_dados,
        prepara_dados=prepara_dados,
        modelos=[
            modelo_regressao_linear,
            modelo_regressao_ridge,
            modelo_regressao_lasso,
            modelo_regressao_elasticnet,
            modelo_regressao_polinomial
        ],
        observadores=[console_obs, mlflow_obs],
    )
    pml.rodar_treinamento_simples()
    # pml.realizar_tuning_parametros()
    # pml.realizar_validacao_cruzada()

