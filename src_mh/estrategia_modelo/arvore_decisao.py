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
from sklearn.tree import DecisionTreeRegressor, export_text, plot_tree

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo

logger = logging.getLogger(__name__)


class ArvoreDecisaoEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):
    """Estratégia Concreta de Machine Learning utilizando Árvore de Decisão Regressora."""

    def __init__(self, params: dict | None = None):
        self.__params = params or {
            "max_depth": getattr(Config, "max_depth_dt", 6),
            "min_samples_split": getattr(Config, "min_samples_split_dt", 5),
            "min_samples_leaf": getattr(Config, "min_samples_leaf_dt", 2),
            "random_state": 42,
        }
        self.__modelo = DecisionTreeRegressor(**self.__params)
        self.__colunas: list[str] = []
        self.__equacoes_por_zona: dict[str, float] = {}
        self.__params_tuning = {
            "max_depth": getattr(Config, "max_depth_dt_turing", [3, 5, 7, 10, None]),
            "min_samples_split": getattr(Config, "min_samples_split_dt_turing", [2, 5, 10]),
            "min_samples_leaf": getattr(Config, "min_samples_leaf_dt_turing", [1, 2, 4]),
        }
        self.__param_range = np.arange(1, 16)

    @property
    @override
    def nome(self) -> str:
        return "Árvore de Decisão"

    @property
    @override
    def modelo_objeto(self) -> object:
        return self.__modelo

    @override
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.__colunas = list(x_train.columns)
        self.__modelo.fit(x_train, y_train)

        # Calcula o preço médio estimado pela Árvore de Decisão por Zona do Bairro
        preds_treino = self.__modelo.predict(x_train)
        equacoes: dict[str, float] = {}
        zona_cols = [c for c in self.__colunas if c.startswith("Zona_")]

        if zona_cols:
            mask_centro = (x_train[zona_cols] == 0).all(axis=1)
            if mask_centro.any():
                equacoes["Centro/Outros (Baseline)"] = round(float(np.mean(preds_treino[mask_centro])), 2)

            for col in zona_cols:
                nome_zona = col.replace("Zona_", "")
                mask = x_train[col] == 1
                if mask.any():
                    equacoes[nome_zona] = round(float(np.mean(preds_treino[mask])), 2)

        self.__equacoes_por_zona = equacoes

    @override
    def predizer(self, x: pd.DataFrame) -> np.ndarray:
        return self.__modelo.predict(x)

    @override
    def obter_equacoes_por_zona(self) -> dict[str, float]:
        """Preço médio estimado pela Árvore de Decisão para cada Zona do Bairro."""
        return self.__equacoes_por_zona

    @override
    def obter_equacao_reta_geral(self) -> str:
        """Exporta o resumo de preço médio por Zonas do Bairro e as regras textuais da Árvore de Decisão."""
        if not hasattr(self.__modelo, "tree_") or len(self.__colunas) == 0:
            return "Árvore de Decisão não treinada."

        regras = export_text(
            self.__modelo,
            feature_names=self.__colunas,
            max_depth=4,
            show_weights=True,
        )

        resumo_zonas = ["=== PREÇO MÉDIO PREVISTO POR ZONA DO BAIRRO (ÁRVORE DE DECISÃO) ==="]
        if self.__equacoes_por_zona:
            for zona, preco in self.__equacoes_por_zona.items():
                resumo_zonas.append(f"• {zona}: R$ {preco:,.2f}".replace(",", "."))
        else:
            resumo_zonas.append("• Zonas não identificadas explicitamente no conjunto de dados.")

        resumo_str = "\n".join(resumo_zonas)
        return f"{resumo_str}\n\n=== REGRAS DE DECISÃO HIERÁRQUICAS (Profundidade <= 4) ===\n{regras}"

    @override
    def obter_curva_validacao(
        self, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, object]:
        """Calcula as pontuações da curva de validação variando max_depth."""
        train_scores, test_scores = validation_curve(
            DecisionTreeRegressor(
                min_samples_split=self.__modelo.min_samples_split,
                min_samples_leaf=self.__modelo.min_samples_leaf,
                random_state=42,
            ),
            x,
            y,
            param_name="max_depth",
            param_range=self.__param_range,
            cv=5,
            scoring="neg_root_mean_squared_error",
        )
        train_rmse = [round(float(-v), 2) for v in np.mean(train_scores, axis=1)]
        val_rmse = [round(float(-v), 2) for v in np.mean(test_scores, axis=1)]
        param_list = self.__param_range.tolist()

        return {
            "param_name": "max_depth",
            "max_depth_range": param_list,
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
        """Gera a figura Matplotlib com as particularidades da Árvore de Decisão (max_depth)."""
        if not dados:
            return None

        param_name = dados.get("param_name", "max_depth")
        param_range = dados.get("max_depth_range") or dados.get("param_range", [])
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

        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2.5, markersize=7, label="RMSE Treino (Viés / Histórico)", zorder=4)
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
                ax.axvspan(x[0], x_opt_start, color="#ffebee", alpha=0.4, label="Região de Underfitting (Árvore Rasa)", zorder=1)
            ax.axvspan(x_opt_start, x_opt_end, color="#e8f5e9", alpha=0.5, label="Região de Ajuste Ótimo da Árvore", zorder=1)
            if min_val_idx < len(x) - 1:
                ax.axvspan(x_opt_end, x[-1], color="#fff8e1", alpha=0.4, label="Região de Overfitting (Árvore Profunda / Memorização)", zorder=1)

        ax.axvline(best_param, color="#2e7d32", linestyle="--", linewidth=2.2, label=f"Melhor max_depth = {best_param_str}", zorder=4)
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markeredgecolor="#2e7d32", markeredgewidth=2.5, markersize=12, zorder=6)

        if min_val_idx > 0:
            ax.text(x[0], y_val[0] + y_range * 0.08, " [ UNDERFITTING ] \n (Árvore Rasa / Poucas Divisões) ", fontsize=9, fontweight="bold", color="#c62828", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.95), zorder=5)

        ax.annotate(
            f" [ AJUSTE ÓTIMO ] (Níveis de Decisão)\n Melhor max_depth = {best_param_str}\n RMSE Validação ≈ R$ {min_val_rmse:,.0f} ".replace(",", "."),
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
            ax.text(x[-1], (y_tr[-1] + y_val[-1]) / 2.0, " [ OVERFITTING ] \n (Árvore Profunda / Memorização) ", fontsize=9, fontweight="bold", color="#f57f17", va="center", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#fbc02d", alpha=0.95), zorder=5)

        ax.set_title(f"{self.nome} — Diagnóstico Preditivo (Profundidade da Árvore)", fontsize=13.5, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Profundidade Máxima da Árvore)", fontsize=10.5, fontweight="bold", labelpad=10)
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
        """Executa busca em grade de hiperparâmetros (GridSearchCV) para Árvore de Decisão."""
        grid_search = GridSearchCV(
            estimator=DecisionTreeRegressor(random_state=42),
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
        """Executa 1 iteração de K-Fold Cross-Validation para a Árvore de Decisão."""
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

    def plotar_importancia_atributos(self) -> plt.Figure | None:
        """Gera um gráfico de barras horizontais rico e legível com a importância dos atributos (Feature Importances)."""
        if not hasattr(self.__modelo, "feature_importances_") or len(self.__colunas) == 0:
            return None

        importancias = self.__modelo.feature_importances_
        if np.sum(importancias) == 0:
            return None

        # Ordena da maior para a menor importância
        indices = np.argsort(importancias)
        colunas_ord = [self.__colunas[i].replace("_", " ") for i in indices]
        importancias_ord = importancias[indices]
        porcentagens_ord = importancias_ord * 100.0

        plt.style.use("seaborn-v0_8-whitegrid")
        fig, ax = plt.subplots(figsize=(12, max(6, len(self.__colunas) * 0.45)), dpi=300)

        # Paleta de cores em gradiente Teal/Verde Esmeralda
        colors = plt.cm.viridis(np.linspace(0.3, 0.85, len(importancias_ord)))

        bars = ax.barh(colunas_ord, porcentagens_ord, color=colors, edgecolor="#1b5e20", linewidth=1.2, height=0.65)

        # Adiciona rótulos numéricos com porcentagem nas extremidades das barras
        max_val = max(porcentagens_ord) if len(porcentagens_ord) > 0 else 1.0
        for bar, pct, val in zip(bars, porcentagens_ord, importancias_ord):
            if pct > 0.05:
                ax.text(
                    pct + max_val * 0.015,
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{pct:.2f}% ({val:.4f})",
                    va="center",
                    ha="left",
                    fontsize=9.5,
                    fontweight="bold",
                    color="#263238",
                )

        # Destaque para o atributo mais relevante (Top 1 Predictor)
        top_idx = len(importancias_ord) - 1
        top_feature = colunas_ord[top_idx]
        top_pct = porcentagens_ord[top_idx]

        offset_y = -0.5 if top_idx >= 2 else 0.0
        ax.annotate(
            f" [ PRINCIPAL ATRIBUTO ] \n {top_feature}: {top_pct:.2f}% de Impacto ",
            xy=(top_pct, top_idx),
            xytext=(top_pct * 0.65, top_idx + offset_y),
            fontsize=9.5,
            fontweight="bold",
            color="#1b5e20",
            ha="right",
            arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.5", facecolor="#e8f5e9", edgecolor="#2e7d32", linewidth=1.5, alpha=0.95),
            zorder=6,
        )

        ax.set_title(
            f"{self.nome} — Importância Relativa dos Atributos (Feature Importances)",
            fontsize=13.5,
            fontweight="bold",
            pad=15,
        )
        ax.set_xlabel("Importância Relativa (%) — Redução de Impureza (Gini / MSE)", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.set_ylabel("Atributos do Imóvel", fontsize=10.5, fontweight="bold", labelpad=10)
        ax.set_xlim(0, max_val * 1.30)
        ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda val, loc: f"{val:.1f}%"))
        ax.grid(True, linestyle="--", alpha=0.5, axis="x")
        fig.tight_layout()

        return fig

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

        importancias_dict = {}
        if hasattr(self.__modelo, "feature_importances_") and len(self.__colunas) > 0:
            importancias_dict = {
                col: round(float(imp), 4)
                for col, imp in zip(self.__colunas, self.__modelo.feature_importances_)
            }

        profundidade_arvore = (
            int(self.__modelo.get_depth())
            if hasattr(self.__modelo, "get_depth")
            else 0
        )
        numero_folhas = (
            int(self.__modelo.get_n_leaves())
            if hasattr(self.__modelo, "get_n_leaves")
            else 0
        )
        numero_nos = (
            int(self.__modelo.tree_.node_count)
            if hasattr(self.__modelo, "tree_")
            else 0
        )

        regras = ""
        if hasattr(self.__modelo, "tree_") and len(self.__colunas) > 0:
            try:
                regras = export_text(
                    self.__modelo,
                    feature_names=self.__colunas,
                    max_depth=3,
                    show_weights=True,
                )
            except Exception:
                regras = ""

        equacao_geral = (
            f"Árvore de Decisão Regressora (Profundidade={profundidade_arvore}, "
            f"Nós={numero_nos}, Folhas={numero_folhas})"
        )

        fig_importancia = self.plotar_importancia_atributos()
        fig_diagrama = self.plotar_diagrama_arvore()

        return {
            "max_depth_range": self.__param_range.tolist(),
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
            "profundidade_arvore": profundidade_arvore,
            "numero_folhas": numero_folhas,
            "numero_nos": numero_nos,
            "feature_importances": importancias_dict,
            "importancia_atributos": importancias_dict,
            "figura_importancia_atributos": fig_importancia,
            "figura_diagrama_arvore": fig_diagrama,
            "criterio_divisao": str(getattr(self.__modelo, "criterion", "squared_error")),
            "estrategia_divisao": str(getattr(self.__modelo, "splitter", "best")),
            "amostras_minimas_divisao": getattr(self.__modelo, "min_samples_split", 2),
            "amostras_minimas_folha": getattr(self.__modelo, "min_samples_leaf", 1),
            "fracao_peso_minimo_folha": float(getattr(self.__modelo, "min_weight_fraction_leaf", 0.0)),
            "max_features": str(getattr(self.__modelo, "max_features", "None")),
            "max_leaf_nodes": str(getattr(self.__modelo, "max_leaf_nodes", "None")),
            "min_impurity_decrease": float(getattr(self.__modelo, "min_impurity_decrease", 0.0)),
            "ccp_alpha": round(float(getattr(self.__modelo, "ccp_alpha", 0.0)), 4),
            "equacao_geral": equacao_geral,
            "regras_arvore": regras,
        }

    def plotar_diagrama_arvore(self) -> plt.Figure | None:
        """Gera a figura visual da estrutura hierárquica da Árvore de Decisão utilizando plot_tree."""
        if not hasattr(self.__modelo, "tree_") or len(self.__colunas) == 0:
            return None

        plt.style.use("default")
        fig, ax = plt.subplots(figsize=(24, 12), dpi=300)
        plot_tree(
            self.__modelo,
            feature_names=self.__colunas,
            filled=True,
            rounded=True,
            precision=2,
            fontsize=8,
            max_depth=3,
            ax=ax,
        )
        ax.set_title(
            f"{self.nome} — Diagrama Estrutural dos Nós de Decisão (Profundidade <= 3)",
            fontsize=14,
            fontweight="bold",
            pad=15,
        )
        fig.tight_layout()
        return fig
