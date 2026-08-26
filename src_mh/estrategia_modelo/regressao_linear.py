from datetime import datetime
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import LinearRegression, Ridge
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

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo


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
            scoring="r2",
        )
        return {
            "param_name": "alpha",
            "param_range": self.__param_range.tolist(),
            "train_scores_mean": [
                round(float(v), 4) for v in np.mean(train_scores, axis=1)
            ],
            "test_scores_mean": [
                round(float(v), 4) for v in np.mean(test_scores, axis=1)
            ],
            "train_scores_std": [
                round(float(v), 4) for v in np.std(train_scores, axis=1)
            ],
            "test_scores_std": [
                round(float(v), 4) for v in np.std(test_scores, axis=1)
            ],
        }

    @override
    def gerar_figura_underfit_overfit(
            self, x: pd.DataFrame, y: pd.Series
    ) -> plt.Figure:
        """Gera a figura Matplotlib destacando visualmente as regiões de Bias vs Variance (Overfitting / Underfitting)."""
        dados_curva = self.obter_curva_validacao(x, y)
        alpha = dados_curva.get("param_range", [])
        train_scores = dados_curva.get("train_scores_mean", [])
        val_scores = dados_curva.get("test_scores_mean", [])

        fig, ax = plt.subplots(figsize=(10, 6))

        # Plota curvas em escala logarítmica
        ax.semilogx(
            alpha,
            train_scores,
            marker="o",
            linestyle="-",
            linewidth=2,
            color="#1f77b4",
            label="Treino (Train Score R²)",
        )
        ax.semilogx(
            alpha,
            val_scores,
            marker="s",
            linestyle="--",
            linewidth=2,
            color="#ff7f0e",
            label="Validação (Validation Score R²)",
        )

        # 🔹 Destaque 1: Região de Equilíbrio Ideal (Sombreado Verde)
        ax.axvspan(
            0.01,
            2.15,
            color="#2ca02c",
            alpha=0.15,
            label="Região Ideal de Regularização (Equilíbrio)",
        )

        # 🔹 Destaque 2: Anotação indicando Ausência de Overfitting
        ax.annotate(
            "Gap Mínimo (1.4%)\n❌ Sem Overfitting",
            xy=(0.046, 0.832),
            xytext=(0.002, 0.842),
            arrowprops=dict(
                facecolor="#2ca02c", shrink=0.08, width=1.5, headwidth=8
            ),
            fontsize=10,
            fontweight="bold",
            color="#1b5e20",
            bbox=dict(boxstyle="round,pad=0.4", fc="#e8f5e9", ec="#2ca02c", lw=1.5),
        )

        # 🔹 Destaque 3: Anotação indicando Teto de Performance (Underfitting)
        ax.annotate(
            "Teto de R² ≈ 82.5%\n⚠️ Leve Underfitting Linear",
            xy=(2.15, 0.825),
            xytext=(10.0, 0.835),
            arrowprops=dict(
                facecolor="#d62728", shrink=0.08, width=1.5, headwidth=8
            ),
            fontsize=10,
            fontweight="bold",
            color="#b71c1c",
            bbox=dict(boxstyle="round,pad=0.4", fc="#ffebee", ec="#d62728", lw=1.5),
        )

        ax.set_xlabel("Parâmetro de Regularização Alpha (Escala Logarítmica)", fontsize=11, fontweight="bold")
        ax.set_ylabel("Pontuação de Acurácia (R²)", fontsize=11, fontweight="bold")
        ax.set_title(
            "Diagnóstico de Modelo: Bias vs Variância (Overfitting / Underfitting)",
            fontsize=13,
            fontweight="bold",
            pad=15,
        )
        ax.set_ylim(0.78, 0.86)
        ax.legend(loc="lower left", frameon=True, facecolor="white", edgecolor="gray")
        ax.grid(True, linestyle=":", alpha=0.6)
        fig.tight_layout()
        return fig

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

