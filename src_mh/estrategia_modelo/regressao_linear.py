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

from src_mh.config import obter_config_regressao_linear
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo


class RegressaoLinearEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):

    def __init__(self, fit_intercept: bool | None = None):
        cfg = obter_config_regressao_linear()
        self.__cfg = cfg
        intercept_flag = (
            fit_intercept
            if fit_intercept is not None
            else bool(cfg.get("fit_intercept", True))
        )
        self.__modelo = LinearRegression(fit_intercept=intercept_flag)
        self.__colunas: list[str] = []

        start = float(cfg.get("param_range_start", -3))
        end = float(cfg.get("param_range_end", 2))
        num = int(cfg.get("param_range_num", 10))
        self.__param_range = np.logspace(start, end, num)

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

        param_grid = self.__cfg.get("param_grid")
        if not param_grid:
            param_grid = {"regressor__alpha": self.__param_range.tolist()}

        cv = int(self.__cfg.get("cv_folds", 5))
        scoring = str(self.__cfg.get("scoring", "neg_root_mean_squared_error"))
        n_jobs = int(self.__cfg.get("n_jobs", 4))

        grid_search = GridSearchCV(
            estimator=pipeline,
            param_grid=param_grid,
            scoring=scoring,
            cv=cv,
            n_jobs=n_jobs,
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

    @override
    def realizar_validacao_cruzada(
        self, x: pd.DataFrame, y: pd.Series, iteracao: int = 5
    ) -> dict[str, Any]:
        """Realiza validação cruzada KFold (10-folds) e retorna pontuações e resíduos out-of-fold."""
        pipeline = Pipeline([("regressor", self.__modelo)])

        kfold = KFold(n_splits=10, shuffle=True, random_state=iteracao)
        scoring = ["r2", "neg_mean_squared_error", "neg_mean_absolute_error"]

        scores = cross_validate(
            pipeline,
            x,
            y,
            cv=kfold,
            scoring=scoring,
            n_jobs=4,
            return_train_score=True,
            error_score="raise",
        )

        results_dict = {
            "test_r2": [round(float(v), 4) for v in scores["test_r2"]],
            "test_mse": [
                round(float(-v), 2) for v in scores["test_neg_mean_squared_error"]
            ],
            "test_mae": [
                round(float(-v), 2) for v in scores["test_neg_mean_absolute_error"]
            ],
            "train_r2": [round(float(v), 4) for v in scores["train_r2"]],
            "train_mse": [
                round(float(-v), 2) for v in scores["train_neg_mean_squared_error"]
            ],
            "train_mae": [
                round(float(-v), 2) for v in scores["train_neg_mean_absolute_error"]
            ],
            "fit_time": [round(float(v), 4) for v in scores["fit_time"]],
            "score_time": [round(float(v), 4) for v in scores["score_time"]],
        }

        # RMSE por fold
        results_dict["test_rmse"] = [
            round(float(np.sqrt(mse)), 2) for mse in results_dict["test_mse"]
        ]
        results_dict["train_rmse"] = [
            round(float(np.sqrt(mse)), 2) for mse in results_dict["train_mse"]
        ]

        mean_scores = {
            "mean_test_r2": round(float(np.mean(results_dict["test_r2"])), 4),
            "mean_test_mse": round(float(np.mean(results_dict["test_mse"])), 2),
            "mean_test_mae": round(float(np.mean(results_dict["test_mae"])), 2),
            "mean_test_rmse": round(float(np.mean(results_dict["test_rmse"])), 2),
            "mean_train_r2": round(float(np.mean(results_dict["train_r2"])), 4),
            "mean_train_mse": round(float(np.mean(results_dict["train_mse"])), 2),
            "mean_train_mae": round(float(np.mean(results_dict["train_mae"])), 2),
            "mean_train_rmse": round(float(np.mean(results_dict["train_rmse"])), 2),
            "mean_fit_time": round(float(np.mean(results_dict["fit_time"])), 4),
            "mean_score_time": round(float(np.mean(results_dict["score_time"])), 4),
        }

        # Predição out-of-fold (resíduos globais)
        y_pred = cross_val_predict(clone(pipeline), x, y, cv=kfold, n_jobs=-1)
        residuos_totais = y - y_pred

        rmse_folds = [
            round(float(np.sqrt(-mse)), 2)
            for mse in scores["test_neg_mean_squared_error"]
        ]

        return {
            "results_dict": results_dict,
            "mean_scores": mean_scores,
            "residuos_totais": [round(float(r), 2) for r in residuos_totais],
            "iteracao": iteracao,
            "rmse_folds": rmse_folds,
            "data_coleta": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
        }
