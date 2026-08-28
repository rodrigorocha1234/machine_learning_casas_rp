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
    def _plotar_diagnostico_overfitting_underfitting(
        self, dados: dict[str, Any]
    ) -> plt.Figure | None:
        """Gera a figura Matplotlib com as particularidades da Regressão Linear Regularizada."""
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

        # Expande limites Y para criar espaço limpo no topo
        y_min = min(min(y_tr), min(y_val))
        y_max = max(max(y_tr), max(y_val))
        y_range = y_max - y_min
        ax.set_ylim(y_min - y_range * 0.08, y_max + y_range * 0.28)

        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2.5, markersize=7, label="RMSE Treino (Viés Linear)", zorder=4)
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
                ax.axvspan(x[0], x_opt_start, color="#fff8e1", alpha=0.4, label="Região de Overfitting (Sem Penalização)", zorder=1)
            ax.axvspan(x_opt_start, x_opt_end, color="#e8f5e9", alpha=0.5, label="Região de Ajuste Ótimo Linear", zorder=1)
            if min_val_idx < len(x) - 1:
                ax.axvspan(x_opt_end, x[-1], color="#ffebee", alpha=0.4, label="Região de Underfitting (Penalização Excessiva)", zorder=1)

        ax.axvline(best_param, color="#2e7d32", linestyle="--", linewidth=2.2, label=f"Melhor alpha = {best_param_str}", zorder=4)
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markeredgecolor="#2e7d32", markeredgewidth=2.5, markersize=12, zorder=6)

        # Anotações em caixas posicionadas sem sobreposição
        # Callout Overfitting (Esquerda)
        ax.text(x[0], y_tr[0] + y_range * 0.15, " [ OVERFITTING ] \n (Sem Penalização / Coeficientes Livres) ", fontsize=9, fontweight="bold", color="#f57f17", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#fbc02d", alpha=0.95), zorder=5)

        # Callout Ajuste Ótimo (Posicionado à esquerda do ponto de mínimo para não bater na legenda)
        ax.annotate(
            f" [ AJUSTE ÓTIMO ] (Linear Regularizada)\n Melhor alpha = {best_param_str}\n RMSE Validação ≈ R$ {min_val_rmse:,.0f} ".replace(",", "."),
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

        # Callout Underfitting (Posicionado na parte inferior direita)
        ax.text(x[-1], y_tr[-1] + y_range * 0.08, " [ UNDERFITTING ] \n (Penalização Excessiva / Coeficientes Achatados) ", fontsize=9, fontweight="bold", color="#c62828", va="bottom", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.95), zorder=5)

        ax.set_title(f"{self.nome} — Diagnóstico Preditivo (Modelo Linear Regularizado)", fontsize=13.5, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Fator de Penalização — Escala Logarítmica)", fontsize=10.5, fontweight="bold", labelpad=10)
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
