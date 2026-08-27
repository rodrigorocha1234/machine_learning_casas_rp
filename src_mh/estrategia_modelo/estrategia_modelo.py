from abc import ABC, abstractmethod
from datetime import datetime
from io import BytesIO
from typing import Any, Generic, TypeVar

import matplotlib.pyplot as plt
import mlflow
import numpy as np
import pandas as pd
from PIL import Image
from sklearn.base import clone
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import (
    GridSearchCV,
    KFold,
    cross_val_predict,
    cross_validate,
)
from sklearn.pipeline import Pipeline

X_in = TypeVar("X_in")
Y_in = TypeVar("Y_in")
Y_out = TypeVar("Y_out")


class EstrategiaModelo(ABC, Generic[X_in, Y_in, Y_out]):

    @property
    @abstractmethod
    def nome(self) -> str:
        """Nome identificador do modelo."""
        pass

    @property
    def modelo_objeto(self) -> object | None:
        """Retorna o estimador/objeto de modelo treinado subjacente (ex: Sklearn, LightGBM, XGBoost)."""
        return None

    @abstractmethod
    def treinar(self, x_train: X_in, y_train: Y_in) -> None:
        """Treina o modelo com os dados de treino fornecidos."""
        pass

    @abstractmethod
    def predizer(self, x: X_in) -> Y_out:
        """Realiza predições utilizando o modelo treinado."""
        pass

    @abstractmethod
    def obter_resultados(self, x_test: X_in, y_test: Y_in) -> dict[str, object]:
        """Avalia o modelo no conjunto de teste e retorna um dicionário com métricas."""
        pass

    @abstractmethod
    def obter_equacoes_por_zona(self) -> dict[str, float]:
        """Retorna interceptos por zona (opcional para modelos lineares)."""
        return {}

    @abstractmethod
    def obter_equacao_reta_geral(self) -> str:
        """Retorna a equação geral da reta (opcional para modelos lineares)."""
        return ""

    @abstractmethod
    def obter_curva_validacao(self, x: X_in, y: Y_in) -> dict[str, object]:
        """Calcula a curva de validação (opcional para modelos que suportam)."""
        return {}

    def _plotar_diagnostico_overfitting_underfitting(
        self,
        dados: dict[str, Any],
        nome_artefato_mlflow: str = "under_over_figura.png",
    ) -> plt.Figure | None:
        """Método protegido para gerar a figura de Curva de Validação no padrão gráfico idêntico ao modelo de referência:
        - Linha azul com marcadores em círculo: RMSE Treino (Viés)
        - Linha laranja com marcadores em círculo: RMSE Validação (Generalização)
        - Região sombreada azul claro: Gap Treino x Validação (Variância)
        - Linha tracejada vertical: Melhor ponto do parâmetro
        - Marcador destacado no valor de mínimo RMSE
        - Textos "UNDERFITTING (alto viés)", "OVERFITTING (alta variância)", "Mínimo RMSE (CV)..."
        """
        if not dados:
            return None

        param_name = dados.get("param_name", "Parâmetro")
        param_range = (
            dados.get("alpha_range")
            or dados.get("max_depth_range")
            or dados.get("param_range", [])
        )
        train_rmse = dados.get("train_rmse") or dados.get("train_scores_mean", [])
        val_rmse = dados.get("val_rmse") or dados.get("test_scores_mean", [])

        if not param_range or not train_rmse or not val_rmse:
            return None

        x = [float(p) for p in param_range]
        y_tr = [float(v) for v in train_rmse]
        y_val = [float(v) for v in val_rmse]

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, 6.5), dpi=300)

        # Plota as curvas de Treino e Validação
        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2, markersize=6, label="RMSE Treino (Viés)")
        ax.plot(x, y_val, "-o", color="#ff7f0e", linewidth=2, markersize=6, label="RMSE Validação (Generalização)")

        # Sombreamento do Gap (Variância)
        ax.fill_between(
            x,
            y_tr,
            y_val,
            color="#1f77b4",
            alpha=0.18,
            label="Gap Treino × Validação (Variância)",
        )

        # Identifica o melhor valor do parâmetro (Mínimo RMSE em Validação)
        min_val_idx = int(np.argmin(y_val))
        best_param = x[min_val_idx]
        min_val_rmse = y_val[min_val_idx]

        best_param_str = f"{best_param:.1f}" if isinstance(best_param, float) else f"{best_param}"
        label_vertical = f"Melhor {param_name} = {best_param_str}"
        ax.axvline(best_param, color="#1f77b4", linestyle="--", linewidth=2, label=label_vertical)

        # Marcador laranja destacado na melhor validação
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markersize=10, zorder=5)

        # Texto Callout de Underfitting (topo esquerdo)
        ax.text(
            x[0],
            max(y_val[0], y_tr[0]),
            "UNDERFITTING\n(alto viés)",
            fontsize=10,
            fontweight="bold",
            va="top",
            ha="left",
        )

        # Texto Callout do Mínimo RMSE
        offset_x = (max(x) - min(x)) * 0.015
        offset_y = (max(y_val) - min(y_tr)) * 0.04
        ax.text(
            best_param + offset_x,
            min_val_rmse + offset_y,
            f"Mínimo RMSE (CV)\nRMSE ≈ {min_val_rmse:,.0f}".replace(",", "."),
            fontsize=9.5,
            fontweight="bold",
            va="bottom",
            ha="left",
        )

        # Texto Callout de Overfitting (canto inferior direito)
        ax.text(
            x[-1],
            (y_tr[-1] + y_val[-1]) / 2.0,
            "OVERFITTING\n(alta variância)",
            fontsize=10,
            fontweight="bold",
            va="center",
            ha="right",
        )

        # Título, rótulos e legendas
        ax.set_title(f"{self.nome} — Diagnóstico de Overfitting vs Underfitting", fontsize=14, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Complexidade do Modelo)", fontsize=11, fontweight="bold")
        ax.set_ylabel("RMSE", fontsize=11, fontweight="bold")

        ax.legend(loc="upper right", frameon=True, facecolor="white", framealpha=0.9)
        ax.grid(True, linestyle="-", alpha=0.7)
        fig.tight_layout()

        return fig

    def gerar_figura_underfit_overfit(self, dados: dict[str, Any]) -> plt.Figure | None:
        """Gera o objeto plt.Figure da curva de validação utilizando a implementação protegida da classe base."""
        return self._plotar_diagnostico_overfitting_underfitting(dados=dados)

    @abstractmethod
    def realizar_grid_search(
            self, x: X_in, y: Y_in
    ) -> GridSearchCV:
        """Realiza busca em grade de hiperparâmetros (GridSearchCV)."""
        return GridSearchCV(estimator=None, param_grid={})

    @abstractmethod
    def obter_resultado_grid_search(
            self, grid_search: GridSearchCV
    ) -> dict[str, Any]:
        """Extrai e estrutura os resultados detalhados da busca em grade (GridSearchCV)."""
        return {}

    def realizar_validacao_cruzada(
            self, x: pd.DataFrame, y: pd.Series, iteracao: int = 5
    ) -> dict[str, Any]:
        """Realiza validação cruzada KFold (10-folds) e retorna pontuações e resíduos out-of-fold."""
        modelo_est = self.modelo_objeto
        if modelo_est is None:
            raise ValueError(f"O modelo '{self.nome}' não possui um estimador válido em modelo_objeto.")

        pipeline = Pipeline([("regressor", modelo_est)])

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

        # Predição out-of-fold (resíduos globais e métricas no conjunto completo)
        y_pred = cross_val_predict(clone(pipeline), x, y, cv=kfold, n_jobs=-1)
        residuos_totais = y - y_pred

        oof_r2 = round(float(r2_score(y, y_pred)), 4)
        oof_rmse = round(float(np.sqrt(mean_squared_error(y, y_pred))), 2)
        oof_mae = round(float(mean_absolute_error(y, y_pred)), 2)

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
            "oof_completo_r2": oof_r2,
            "oof_completo_rmse": oof_rmse,
            "oof_completo_mae": oof_mae,
        }

        rmse_folds = [
            round(float(np.sqrt(-mse)), 2)
            for mse in scores["test_neg_mean_squared_error"]
        ]

        # Ajusta o modelo final da iteração e extrai amostra para registro do modelo no MLflow
        pipeline_fitted = clone(pipeline).fit(x, y)
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
