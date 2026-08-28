import logging
from datetime import datetime
from io import BytesIO
from typing import Any, override

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.linear_model import Lasso
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_validate,
    validation_curve,
)

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo

logger = logging.getLogger(__name__)


class RegressaoLassoEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):
    """Estratégia Concreta de Machine Learning utilizando Regressão Lasso (Regularização L1)."""

    def __init__(self, params: dict | None = None):
        self.__params = params or {
            "alpha": getattr(Config, "alpha_lasso", 1.0),
            "fit_intercept": getattr(Config, "fit_intercept_lasso", True),
            "max_iter": getattr(Config, "max_iter_lasso", 15000),
            "tol": getattr(Config, "tol_lasso", 0.001),
            "random_state": 42,
        }
        self.__modelo = Lasso(**self.__params)
        self.__colunas: list[str] = []
        self.__params_tuning = {
            "alpha": getattr(Config, "alpha_lasso_turing", [0.01, 0.1, 1.0, 10.0, 100.0]),
            "fit_intercept": getattr(Config, "fit_intercept_turing_lasso", [True, False]),
            "selection": getattr(Config, "selection_turing_lasso", ["cyclic", "random"]),
        }

        start = getattr(Config, "param_range_start_rl", -3)
        end = getattr(Config, "param_range_end_rl", 2)
        num = getattr(Config, "param_range_num_rl", 10)
        self.__param_range = np.logspace(start, end, num)

    @property
    @override
    def nome(self) -> str:
        return "Regressão Lasso"

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
        """Gera a representação em texto da equação geral da reta de regressão Lasso."""
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
        """Calcula as pontuações da curva de validação variando alpha."""
        train_scores, test_scores = validation_curve(
            Lasso(
                fit_intercept=self.__modelo.fit_intercept,
                max_iter=getattr(Config, "max_iter_lasso", 15000),
                tol=getattr(Config, "tol_lasso", 0.001),
                random_state=42,
            ),
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
    def _plotar_diagnostico_overfitting_underfitting(
        self, dados: dict[str, Any]
    ) -> plt.Figure | None:
        """Gera a figura Matplotlib com as particularidades da Regressão Lasso (Penalização L1 / Zeramento de Coeficientes)."""
        if not dados:
            return None

        param_name = dados.get("param_name", "alpha")
        param_range = dados.get("alpha_range") or dados.get("param_range", [])
        train_rmse = dados.get("train_rmse") or dados.get("train_scores_mean", [])
        val_rmse = dados.get("val_rmse") or dados.get("test_scores_mean", [])

        if not param_range or not train_rmse or not val_rmse:
            return None

        x = [float(p) for p in param_range]
        y_tr = [float(v) for v in train_rmse]
        y_val = [float(v) for v in val_rmse]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(13, 7.5), dpi=300)
        if min(x) > 0:
            ax.set_xscale("log")

        y_min = min(min(y_tr), min(y_val))
        y_max = max(max(y_tr), max(y_val))
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.28)

        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2.5, markersize=7, label="RMSE Treino (Viés L1)", zorder=4)
        ax.plot(x, y_val, "-o", color="#ff7f0e", linewidth=2.5, markersize=7, label="RMSE Validação (Generalização)", zorder=4)
        ax.fill_between(x, y_tr, y_val, color="#1f77b4", alpha=0.15, label="Gap Treino × Validação (Variância)", zorder=2)

        min_val_idx = int(np.argmin(y_val))
        best_param = x[min_val_idx]
        min_val_rmse = y_val[min_val_idx]
        best_param_str = f"{best_param:.4f}" if best_param < 0.01 else f"{best_param:.2f}"

        if len(x) > 1:
            x_opt_start = x[max(0, min_val_idx - 1)] if min_val_idx > 0 else x[0]
            x_opt_end = x[min(len(x) - 1, min_val_idx + 1)] if min_val_idx < len(x) - 1 else x[-1]
            if min_val_idx > 0:
                ax.axvspan(x[0], x_opt_start, color="#fff8e1", alpha=0.4, label="Região de Overfitting (Sem Penalização L1)", zorder=1)
            ax.axvspan(x_opt_start, x_opt_end, color="#e8f5e9", alpha=0.5, label="Região de Ajuste Ótimo Lasso", zorder=1)
            if min_val_idx < len(x) - 1:
                ax.axvspan(x_opt_end, x[-1], color="#ffebee", alpha=0.4, label="Região de Underfitting (Zeramento Excessivo L1)", zorder=1)

        ax.axvline(best_param, color="#2e7d32", linestyle="--", linewidth=2.2, label=f"Melhor alpha L1 = {best_param_str}", zorder=4)
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markeredgecolor="#2e7d32", markeredgewidth=2.5, markersize=12, zorder=6)

        ax.text(x[0], y_tr[0] + y_range * 0.15, " [ OVERFITTING ] \n (Sem Penalização L1 / Coeficientes Livres) ", fontsize=9, fontweight="bold", color="#f57f17", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#fbc02d", alpha=0.95), zorder=5)
        ax.annotate(
            f" [ AJUSTE ÓTIMO ] (Penalização L1 - Seleção)\n Melhor alpha = {best_param_str}\n RMSE Validação ≈ R$ {min_val_rmse:,.0f} ".replace(",", "."),
            xy=(best_param, min_val_rmse),
            xytext=(best_param * 0.04, min_val_rmse + y_range * 0.10),
            fontsize=9,
            fontweight="bold",
            color="#1b5e20",
            ha="center",
            arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#2e7d32", linewidth=1.5, alpha=0.95),
            zorder=6,
        )
        ax.text(x[-1], y_tr[-1] + y_range * 0.08, " [ UNDERFITTING ] \n (Penalização L1 Excessiva / Atributos Zerados) ", fontsize=9, fontweight="bold", color="#c62828", va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.95), zorder=5)

        ax.set_title(f"{self.nome} — Diagnóstico Preditivo (Penalização L1 - Lasso)", fontsize=13.5, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Fator de Penalização L1 — Escala Logarítmica)", fontsize=10.5, fontweight="bold", labelpad=10)
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
        """Executa busca em grade de hiperparâmetros (GridSearchCV) para Regressão Lasso."""
        grid_search = GridSearchCV(
            estimator=Lasso(max_iter=2000),
            param_grid=self.__params_tuning,
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
    def realizar_validacao_cruzada(
        self, x: pd.DataFrame, y: pd.Series, iteracao: int = 0
    ) -> dict[str, Any]:
        """Executa 1 iteração de K-Fold Cross-Validation para a estratégia Lasso."""
        kf = KFold(n_splits=5, shuffle=True, random_state=iteracao)

        scoring = {
            "r2": "r2",
            "neg_mean_squared_error": "neg_mean_squared_error",
            "neg_mean_absolute_error": "neg_mean_absolute_error",
        }

        scores = cross_validate(
            self.__modelo,
            x,
            y,
            cv=kf,
            scoring=scoring,
            return_train_score=True,
            n_jobs=-1,
        )

        oof_pred = np.zeros(len(x))
        residuos_totais: list[float] = []

        for train_idx, val_idx in kf.split(x, y):
            x_tr, y_tr = x.iloc[train_idx], y.iloc[train_idx]
            x_val, y_val = x.iloc[val_idx], y.iloc[val_idx]

            mdl = clone(self.__modelo)
            mdl.fit(x_tr, y_tr)
            preds = mdl.predict(x_val)
            oof_pred[val_idx] = preds
            residuos_totais.extend((y_val - preds).tolist())

        y_arr = np.asarray(y, dtype=float)
        oof_r2 = round(float(r2_score(y_arr, oof_pred)), 4)
        oof_rmse = round(float(np.sqrt(mean_squared_error(y_arr, oof_pred))), 2)
        oof_mae = round(float(mean_absolute_error(y_arr, oof_pred)), 2)

        results_dict = {
            "test_r2": [round(float(v), 4) for v in scores["test_r2"]],
            "test_rmse": [round(float(np.sqrt(-v)), 2) for v in scores["test_neg_mean_squared_error"]],
            "test_mae": [round(float(-v), 2) for v in scores["test_neg_mean_absolute_error"]],
            "train_r2": [round(float(v), 4) for v in scores["train_r2"]],
            "train_rmse": [round(float(np.sqrt(-v)), 2) for v in scores["train_neg_mean_squared_error"]],
            "train_mae": [round(float(-v), 2) for v in scores["train_neg_mean_absolute_error"]],
            "fit_time": [round(float(v), 4) for v in scores["fit_time"]],
            "score_time": [round(float(v), 4) for v in scores["score_time"]],
        }

        mean_scores = {
            "mean_test_r2": round(float(np.mean(results_dict["test_r2"])), 4),
            "mean_test_mse": round(float(np.mean([-v for v in scores["test_neg_mean_squared_error"]])), 2),
            "mean_test_mae": round(float(np.mean(results_dict["test_mae"])), 2),
            "mean_test_rmse": round(float(np.mean(results_dict["test_rmse"])), 2),
            "mean_train_r2": round(float(np.mean(results_dict["train_r2"])), 4),
            "mean_train_mse": round(float(np.mean([-v for v in scores["train_neg_mean_squared_error"]])), 2),
            "mean_train_mae": round(float(np.mean(results_dict["train_mae"])), 2),
            "mean_train_rmse": round(float(np.mean(results_dict["train_rmse"])), 2),
            "mean_fit_time": round(float(np.mean(results_dict["fit_time"])), 4),
            "mean_score_time": round(float(np.mean(results_dict["score_time"])), 4),
            "oof_completo_r2": oof_r2,
            "oof_completo_rmse": oof_rmse,
            "oof_completo_mae": oof_mae,
        }

        rmse_folds = [
            round(float(np.sqrt(-mse)), 2)
            for mse in scores["test_neg_mean_squared_error"]
        ]

        pipeline_fitted = clone(self.__modelo).fit(x, y)
        x_sample = x.head(5)
        y_sample = pipeline_fitted.predict(x_sample)

        return {
            "results_dict": results_dict,
            "mean_scores": mean_scores,
            "residuos_totais": [round(float(r), 2) for r in residuos_totais],
            "iteracao": iteracao,
            "rmse_folds": rmse_folds,
            "data_coleta": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
            "modelo_objeto": pipeline_fitted,
            "x_sample": x_sample,
            "y_sample": y_sample,
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
            "intercepto": round(float(self.__modelo.intercept_), 2),
            "coeficientes": coeficientes_dict,
        }
