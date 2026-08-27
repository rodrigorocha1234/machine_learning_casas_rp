from datetime import datetime
from io import BytesIO
import logging
from typing import Any, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import LinearRegression
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_validate,
    validation_curve,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo

logger = logging.getLogger(__name__)


class RegressaoPolinomialEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):
    """Estratégia de Regressão Polinomial integrada à arquitetura EstrategiaModelo."""

    def __init__(self, params: dict | None = None):
        self._logger = logger
        self.__params = params or {}

        degree = int(self.__params.get("degree", Config.degree_rp))
        include_bias = bool(self.__params.get("include_bias", Config.include_bias_rp))
        fit_intercept = bool(self.__params.get("fit_intercept", Config.fit_intercept_rp))
        positive = bool(self.__params.get("positive", Config.positive_rp))

        self.__modelo = Pipeline([
            ("poly", PolynomialFeatures(degree=degree, include_bias=include_bias)),
            ("regressor", LinearRegression(fit_intercept=fit_intercept, positive=positive)),
        ])

        self.__params_turing = {
            "poly__degree": Config.poly_degree_turing_rp,
            "regressor__fit_intercept": Config.fit_intercept_turing_rp,
            "regressor__positive": Config.positive_turing_rp,
        }

    @property
    @override
    def nome(self) -> str:
        return "Regressão Polinomial"

    @property
    @override
    def modelo_objeto(self) -> object:
        return self.__modelo

    @override
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.__modelo.fit(x_train, y_train)

    @override
    def predizer(self, x: pd.DataFrame) -> np.ndarray:
        return self.__modelo.predict(x)

    @override
    def obter_resultados(
        self, x_test: pd.DataFrame, y_test: pd.Series
    ) -> dict[str, object]:
        y_pred = self.predizer(x_test)
        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)

        return {
            "r2_score": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 2),
            "mse": round(float(mse), 2),
            "rmse": round(float(rmse), 2),
            "median_ae": round(float(median_absolute_error(y_test, y_pred)), 2),
        }

    @override
    def obter_equacoes_por_zona(self) -> dict[str, float]:
        """Retorna interceptos por zona (opcional para modelos lineares simples)."""
        return {}

    @override
    def obter_equacao_reta_geral(self) -> str:
        """Retorna a representação textual da equação do modelo."""
        return ""

    @override
    def obter_curva_validacao(
        self, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, object]:
        """Calcula as pontuações da curva de validação variando o grau polinomial."""
        param_range = [1, 2, 3]
        pipeline_cv = Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("regressor", LinearRegression()),
        ])

        train_scores, test_scores = validation_curve(
            pipeline_cv,
            x,
            y,
            param_name="poly__degree",
            param_range=param_range,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        train_rmse = [round(float(-v), 2) for v in np.mean(train_scores, axis=1)]
        val_rmse = [round(float(-v), 2) for v in np.mean(test_scores, axis=1)]

        return {
            "param_name": "poly__degree",
            "alpha_range": param_range,
            "param_range": param_range,
            "train_rmse": train_rmse,
            "val_rmse": val_rmse,
            "train_scores_mean": train_rmse,
            "test_scores_mean": val_rmse,
        }

    @override
    def gerar_figura_underfit_overfit(
        self, dados: dict[str, Any]
    ) -> plt.Figure | None:
        """Gera a figura de Diagnóstico de Overfitting vs Underfitting no padrão gráfico da classe base."""
        return self._plotar_diagnostico_overfitting_underfitting(
            dados=dados,
            nome_artefato_mlflow="under_over_polinomial.png",
        )

    @override
    def realizar_grid_search(
        self, x: pd.DataFrame, y: pd.Series
    ) -> GridSearchCV:
        """Executa busca em grade de hiperparâmetros (GridSearchCV) para a Regressão Polinomial."""
        pipeline_search = Pipeline([
            ("poly", PolynomialFeatures(include_bias=False)),
            ("regressor", LinearRegression()),
        ])

        grid_search = GridSearchCV(
            estimator=pipeline_search,
            param_grid=self.__params_turing,
            scoring="neg_root_mean_squared_error",
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
