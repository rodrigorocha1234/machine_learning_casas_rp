import logging
from io import BytesIO
from typing import Any, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    validation_curve,
)
from sklearn.pipeline import Pipeline

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo

logger = logging.getLogger(__name__)


class RegressaoLinearEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):

    def __init__(self, params: dict):

        self.__params = params
        self.__modelo = LinearRegression(**params)
        self.__colunas: list[str] = []
        self.__params_turing = {
            'regressor__fit_intercept': Config.fit_intercept_turing_rl,
            'regressor__positive': Config.positive_turing_rl
        }

        start = Config.param_range_start_rl
        end = Config.param_range_end_rl
        num = Config.param_range_num_rl
        self.__param_range = np.logspace(start, end, num)
        self.__regressor = Pipeline(
            [
                (
                    "regressor",
                    LinearRegression()
                )
            ]
        )

    @property
    @override
    def nome(self) -> str:
        return "Regressão Linear"

    @property
    @override
    def modelo_objeto(self) -> object:
        return self.__modelo

    @override
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.__colunas = list(x_train.columns)
        self.__modelo.fit(x_train, y_train)

    @override
    def predizer(self, x: pd.DataFrame) -> np.ndarray:
        return self.__modelo.predict(x)

    @override
    def obter_equacoes_por_zona(self) -> dict[str, float]:
        """Calcula o intercepto efetivo ajustado para cada zona da cidade."""
        intercepto_base = float(self.__modelo.intercept_)
        equacoes: dict[str, float] = {
            "Centro/Outros (Baseline)": round(intercepto_base, 2)
        }

        for col, coef in zip(self.__colunas, self.__modelo.coef_):
            if col.startswith("Zona_"):
                nome_zona = col.replace("Zona_", "")
                equacoes[nome_zona] = round(float(intercepto_base + coef), 2)

        return equacoes

    @override
    def obter_equacao_reta_geral(self) -> str:
        """Gera a representação em texto da equação geral unificada da reta de regressão."""
        intercepto_base = round(float(self.__modelo.intercept_), 2)
        termos = [f"{intercepto_base}", "+ (0.00 * Zona_Centro)"]

        for col, coef in zip(self.__colunas, self.__modelo.coef_):
            coef_r = round(float(coef), 2)
            sinal = "+" if coef_r >= 0 else "-"
            termos.append(f"{sinal} ({abs(coef_r)} * {col})")

        return "Valor_da_Venda = " + " ".join(termos)

    @override
    def obter_curva_validacao(
            self, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, object]:
        """Calcula as pontuações da curva de validação (validation_curve)."""
        train_scores, test_scores = validation_curve(
            Ridge(fit_intercept=self.__modelo.fit_intercept),
            x,
            y,
            param_name="alpha",
            param_range=self.__param_range,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        train_rmse = [round(float(-v), 2) for v in np.mean(train_scores, axis=1)]
        val_rmse = [round(float(-v), 2) for v in np.mean(test_scores, axis=1)]
        param_list = self.__param_range.tolist()

        return {
            "param_name": "alpha",
            "alpha_range": param_list,
            "param_range": param_list,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_scores_mean": train_rmse,
            "test_scores_mean": val_rmse,
        }

    @override
    def gerar_figura_underfit_overfit(self, dados: dict[str, Any]) -> plt.Figure | None:
        """Gera a figura de Diagnóstico de Overfitting vs Underfitting no padrão gráfico da classe base."""
        return self._plotar_diagnostico_overfitting_underfitting(
            dados=dados,
            nome_artefato_mlflow="under_over_linear.png",
        )

    @override
    def realizar_grid_search(
            self, x: pd.DataFrame, y: pd.Series
    ) -> GridSearchCV:
        """Executa busca em grade de hiperparâmetros (GridSearchCV) utilizando as configurações do YAML/Config."""
        pipeline = Pipeline([("regressor", Ridge(fit_intercept=self.__modelo.fit_intercept))])

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=self.__params_turing,
            scoring='neg_root_mean_squared_error',
            cv=5,
            n_jobs=4,
            verbose=1,
            return_train_score=True,
        )

        grid_search.fit(x, y)
        return grid_search

    @override
    def obter_resultado_grid_search(
            self, grid_search: GridSearchCV
    ) -> dict[str, Any]:
        """Extrai e estrutura os resultados detalhados da busca em grade (GridSearchCV)."""
        if not hasattr(grid_search, "best_params_"):
            return {}

        best_index = int(grid_search.best_index_)
        cv_results = grid_search.cv_results_

        return {
            "melhores_parametros": grid_search.best_params_,
            "melhor_score_cv": round(float(grid_search.best_score_), 4),
            "score_medio_treino": round(
                float(cv_results["mean_test_score"][best_index]), 4
            ),
            "desvio_padrao_cv": round(
                float(cv_results["std_test_score"][best_index]), 4
            ),
            "n_splits": int(grid_search.n_splits_),
            "melhor_estimador": str(grid_search.best_estimator_),
        }

    @override
    def obter_resultados(
            self, x_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, object]:
        y_pred = self.predizer(x_test)
        y_test_arr = np.asarray(y_test, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        mse = mean_squared_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        smape = 100 * np.mean(
            2
            * np.abs(y_pred_arr - y_test_arr)
            / (np.abs(y_test_arr) + np.abs(y_pred_arr) + 1e-8)
        )
        bias = np.mean(y_pred_arr - y_test_arr)
        acc_10 = np.mean(
            np.abs(y_pred_arr - y_test_arr)
            / np.maximum(np.abs(y_test_arr), 1e-8)
            <= 0.10
        )

        coeficientes_dict = {
            col: round(float(coef), 2)
            for col, coef in zip(self.__colunas, self.__modelo.coef_)
        }

        return {
            "alpha_range": self.__param_range.tolist(),
            "mae": round(float(mae), 2),
            "rmse": round(float(rmse), 2),
            "medae": round(float(medae), 2),
            "smape": round(float(smape), 4),
            "r2": round(float(r2), 4),
            "bias": round(float(bias), 2),
            "accuracy_erro_10_pct": round(float(acc_10), 4),
            "preco_medio_real": round(float(np.mean(y_test_arr)), 2),
            "preco_medio_previsto": round(float(np.mean(y_pred_arr)), 2),
            "n_amostras": int(len(y_test_arr)),
            # Interpretabilidade
            "intercepto": round(float(self.__modelo.intercept_), 2),
            "coeficientes": coeficientes_dict,
        }
