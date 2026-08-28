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
    def _plotar_diagnostico_overfitting_underfitting(
        self, dados: dict[str, Any]
    ) -> plt.Figure | None:
        """Gera a figura Matplotlib com as particularidades da Regressão Polinomial (poly__degree)."""
        if not dados:
            return None

        param_name = dados.get("param_name", "poly__degree")
        param_range = dados.get("degree_range") or dados.get("param_range", [])
        train_rmse = dados.get("train_rmse") or dados.get("train_scores_mean", [])
        val_rmse = dados.get("val_rmse") or dados.get("test_scores_mean", [])

        if not param_range or not train_rmse or not val_rmse:
            return None

        x = [float(p) for p in param_range]
        y_tr = [float(v) for v in train_rmse]
        y_val = [float(v) for v in val_rmse]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)

        y_min = min(min(y_tr), min(y_val))
        y_max = max(max(y_tr), max(y_val))
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.28)

        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2.5, markersize=7, label="RMSE Treino (Viés Polinomial)", zorder=4)
        ax.plot(x, y_val, "-o", color="#ff7f0e", linewidth=2.5, markersize=7, label="RMSE Validação (Generalização)", zorder=4)
        ax.fill_between(x, y_tr, y_val, color="#1f77b4", alpha=0.15, label="Gap Treino × Validação (Variância)", zorder=2)

        min_val_idx = int(np.argmin(y_val))
        best_param = x[min_val_idx]
        min_val_rmse = y_val[min_val_idx]
        best_param_str = f"{int(best_param)}" if isinstance(best_param, float) and best_param.is_integer() else f"{best_param}"

        if len(x) > 1:
            x_opt_start = x[max(0, min_val_idx - 1)] if min_val_idx > 0 else x[0]
            x_opt_end = x[min(len(x) - 1, min_val_idx + 1)] if min_val_idx < len(x) - 1 else x[-1]
            if min_val_idx > 0:
                ax.axvspan(x[0], x_opt_start, color="#ffebee", alpha=0.4, label="Região de Underfitting (Reta Rígida / Grau 1)", zorder=1)
            ax.axvspan(x_opt_start, x_opt_end, color="#e8f5e9", alpha=0.5, label="Região de Ajuste Ótimo Polinomial", zorder=1)
            if min_val_idx < len(x) - 1:
                ax.axvspan(x_opt_end, x[-1], color="#fff8e1", alpha=0.4, label="Região de Overfitting (Grau Elevado / Oscilação)", zorder=1)

        ax.axvline(best_param, color="#2e7d32", linestyle="--", linewidth=2.2, label=f"Melhor grau = {best_param_str}", zorder=4)
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markeredgecolor="#2e7d32", markeredgewidth=2.5, markersize=12, zorder=6)

        if min_val_idx > 0:
            ax.text(x[0], y_val[0] + y_range * 0.08, " [ UNDERFITTING ] \n (Grau Linear / Hipótese Rígida) ", fontsize=9, fontweight="bold", color="#c62828", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.95), zorder=5)

        ax.annotate(
            f" [ AJUSTE ÓTIMO ] (Curvatura das Variáveis)\n Melhor grau = {best_param_str}\n RMSE Validação ≈ R$ {min_val_rmse:,.0f} ".replace(",", "."),
            xy=(best_param, min_val_rmse),
            xytext=(best_param, min_val_rmse + y_range * 0.12),
            fontsize=9,
            fontweight="bold",
            color="#1b5e20",
            ha="center",
            arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#2e7d32", linewidth=1.5, alpha=0.95),
            zorder=6,
        )

        if len(x) > 1 and min_val_idx < len(x) - 1:
            ax.text(x[-1], (y_tr[-1] + y_val[-1]) / 2.0, " [ OVERFITTING ] \n (Grau Elevado / Oscilação) ", fontsize=9, fontweight="bold", color="#f57f17", va="center", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#fbc02d", alpha=0.95), zorder=5)

        ax.set_title(f"{self.nome} — Diagnóstico Preditivo (Grau Polinomial)", fontsize=13.5, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Grau de Curvatura do Polinômio)", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.set_ylabel("Erro Preditivo (RMSE em R$)", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"R$ {val:,.0f}".replace(",", ".")))
        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.95, fontsize=8.5, labelspacing=0.4)
        ax.grid(True, linestyle="--", alpha=0.5)
        fig.tight_layout()
        return fig

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
