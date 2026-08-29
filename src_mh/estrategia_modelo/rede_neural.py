import logging
from datetime import datetime
from typing import Any, override

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import TransformedTargetRegressor
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
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src_mh.config.config import Config
from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo

logger = logging.getLogger(__name__)


class RedeNeuralEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):
    """Estratégia Concreta de Machine Learning utilizando Rede Neural Artificial Multi-layer Perceptron (MLPRegressor)."""

    def __init__(self, params: dict | None = None):
        self.__params = params or {
            "hidden_layer_sizes": getattr(Config, "hidden_layer_sizes_nn", (100, 50)),
            "activation": getattr(Config, "activation_nn", "relu"),
            "solver": getattr(Config, "solver_nn", "adam"),
            "alpha": getattr(Config, "alpha_nn", 0.01),
            "learning_rate_init": getattr(Config, "learning_rate_init_nn", 0.01),
            "max_iter": getattr(Config, "max_iter_nn", 1000),
            "early_stopping": getattr(Config, "early_stopping_nn", True),
            "random_state": 42,
        }

        inner_pipeline = Pipeline([
            ("scaler", StandardScaler()),
            ("mlp", MLPRegressor(
                hidden_layer_sizes=self.__params.get("hidden_layer_sizes", (100, 50)),
                activation=self.__params.get("activation", "relu"),
                solver=self.__params.get("solver", "adam"),
                alpha=self.__params.get("alpha", 0.01),
                learning_rate_init=self.__params.get("learning_rate_init", 0.01),
                max_iter=self.__params.get("max_iter", 1000),
                early_stopping=self.__params.get("early_stopping", True),
                random_state=42,
            )),
        ])

        self.__modelo = TransformedTargetRegressor(
            regressor=inner_pipeline,
            transformer=StandardScaler(),
        )

        self.__colunas: list[str] = []
        self.__equacoes_por_zona: dict[str, float] = {}
        self.__params_tuning = {
            "regressor__mlp__hidden_layer_sizes": getattr(Config, "hidden_layer_sizes_nn_turing", [[50], [100, 50], [100, 100]]),
            "regressor__mlp__activation": getattr(Config, "activation_nn_turing", ["relu", "tanh"]),
            "regressor__mlp__alpha": getattr(Config, "alpha_nn_turing", [0.0001, 0.01, 1.0]),
        }
        self.__param_range = np.logspace(-4, 2, 10)  # alpha variando de 0.0001 a 100.0

    @property
    @override
    def nome(self) -> str:
        return "Rede Neural"

    @property
    @override
    def modelo_objeto(self) -> object:
        return self.__modelo

    @override
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.__colunas = list(x_train.columns)
        self.__modelo.fit(x_train, y_train)

        # Calcula o preço médio estimado pela Rede Neural por Zona do Bairro
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
        """Preço médio estimado pela Rede Neural para cada Zona do Bairro."""
        return self.__equacoes_por_zona

    def __obter_mlp_estimator(self) -> MLPRegressor:
        """Recupera a instância do estimador MLPRegressor dentro do TransformedTargetRegressor e Pipeline."""
        if hasattr(self.__modelo, "regressor_") and hasattr(self.__modelo.regressor_, "named_steps"):
            return self.__modelo.regressor_.named_steps.get("mlp", self.__modelo.regressor_)
        elif hasattr(self.__modelo, "regressor") and hasattr(self.__modelo.regressor, "named_steps"):
            return self.__modelo.regressor.named_steps.get("mlp", self.__modelo.regressor)
        return MLPRegressor()

    @override
    def obter_equacao_reta_geral(self) -> str:
        """Exporta a descrição da arquitetura e das previsões por Zona da Rede Neural Artificial."""
        mlp_obj = self.__obter_mlp_estimator()
        camadas = str(getattr(mlp_obj, "hidden_layer_sizes", self.__params.get("hidden_layer_sizes", (100, 50))))
        ativacao = str(getattr(mlp_obj, "activation", self.__params.get("activation", "relu")))
        otimizador = str(getattr(mlp_obj, "solver", self.__params.get("solver", "adam")))
        alpha_val = getattr(mlp_obj, "alpha", self.__params.get("alpha", 0.01))
        n_iter = int(getattr(mlp_obj, "n_iter_", 0))
        loss_val = round(float(getattr(mlp_obj, "loss_", 0.0)), 6)

        resumo_zonas = ["=== PREÇO MÉDIO PREVISTO POR ZONA DO BAIRRO (REDE NEURAL) ==="]
        if self.__equacoes_por_zona:
            for zona, preco in self.__equacoes_por_zona.items():
                resumo_zonas.append(f"• {zona}: R$ {preco:,.2f}".replace(",", "."))
        else:
            resumo_zonas.append("• Zonas não identificadas no conjunto de dados.")

        resumo_str = "\n".join(resumo_zonas)
        return (
            f"Rede Neural Multi-layer Perceptron (MLPRegressor)\n"
            f"• Topologia das Camadas Ocultas: {camadas}\n"
            f"• Função de Ativação: {ativacao.upper()} | Otimizador: {otimizador.upper()}\n"
            f"• Regularização L2 (Alpha): {alpha_val} | Épocas Executadas: {n_iter} | Loss Final: {loss_val}\n\n"
            f"{resumo_str}"
        )

    @override
    def obter_curva_validacao(
        self, x: pd.DataFrame, y: pd.Series
    ) -> dict[str, object]:
        """Calcula as pontuações da curva de validação variando o parâmetro de regularização L2 alpha."""
        base_estimator = TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(
                    hidden_layer_sizes=self.__params.get("hidden_layer_sizes", (100, 50)),
                    activation=self.__params.get("activation", "relu"),
                    solver=self.__params.get("solver", "adam"),
                    learning_rate_init=self.__params.get("learning_rate_init", 0.01),
                    max_iter=500,
                    early_stopping=True,
                    random_state=42,
                )),
            ]),
            transformer=StandardScaler(),
        )

        train_scores, test_scores = validation_curve(
            base_estimator,
            x,
            y,
            param_name="regressor__mlp__alpha",
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
        """Gera a figura Matplotlib com as particularidades da Rede Neural (Regularização L2 / Alpha)."""
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

        ax.plot(x, y_tr, "-o", color="#1f77b4", linewidth=2.5, markersize=7, label="RMSE Treino (Viés Rede Neural)", zorder=4)
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
                ax.axvspan(x[0], x_opt_start, color="#fff8e1", alpha=0.4, label="Região de Overfitting (Alpha Baixo / Pesos Livres)", zorder=1)
            ax.axvspan(x_opt_start, x_opt_end, color="#e8f5e9", alpha=0.5, label="Região de Ajuste Ótimo MLP", zorder=1)
            if min_val_idx < len(x) - 1:
                ax.axvspan(x_opt_end, x[-1], color="#ffebee", alpha=0.4, label="Região de Underfitting (Penalização L2 Excessiva)", zorder=1)

        ax.axvline(best_param, color="#2e7d32", linestyle="--", linewidth=2.2, label=f"Melhor alpha = {best_param_str}", zorder=4)
        ax.plot(best_param, min_val_rmse, "o", color="#ff7f0e", markeredgecolor="#2e7d32", markeredgewidth=2.5, markersize=12, zorder=6)

        if min_val_idx > 0:
            ax.text(x[0], y_val[0] + y_range * 0.08, " [ OVERFITTING ] \n (Alpha Mínimo / Alta Sensibilidade a Ruídos) ", fontsize=9, fontweight="bold", color="#f57f17", va="bottom", ha="left", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#fbc02d", alpha=0.95), zorder=5)

        ax.annotate(
            f" [ AJUSTE ÓTIMO ] (Rede Neural MLP)\n Melhor alpha = {best_param_str}\n RMSE Validação ≈ R$ {min_val_rmse:,.0f} ".replace(",", "."),
            xy=(best_param, min_val_rmse),
            xytext=(best_param * 0.05 if best_param > 0.01 else best_param * 5.0, min_val_rmse + y_range * 0.12),
            fontsize=9,
            fontweight="bold",
            color="#1b5e20",
            ha="center",
            arrowprops=dict(facecolor="#2e7d32", shrink=0.08, width=1.5, headwidth=6),
            bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#2e7d32", linewidth=1.5, alpha=0.95),
            zorder=6,
        )

        if len(x) > 1 and min_val_idx < len(x) - 1:
            ax.text(x[-1], (y_tr[-1] + y_val[-1]) / 2.0, " [ UNDERFITTING ] \n (Alpha Excessivo / Encolhimento de Pesos) ", fontsize=9, fontweight="bold", color="#c62828", va="center", ha="right", bbox=dict(boxstyle="round,pad=0.4", facecolor="#ffffff", edgecolor="#e57373", alpha=0.95), zorder=5)

        ax.set_title(f"{self.nome} — Diagnóstico Preditivo (Parâmetro de Regularização L2 - Alpha)", fontsize=13.5, fontweight="bold", pad=15)
        ax.set_xlabel(f"{param_name} (Parâmetro de Regularização L2 — Escala Logarítmica)", fontsize=10.5, fontweight="bold", labelpad=10)
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
        """Executa busca em grade de hiperparâmetros (GridSearchCV) para a Rede Neural."""
        base_pipeline = TransformedTargetRegressor(
            regressor=Pipeline([
                ("scaler", StandardScaler()),
                ("mlp", MLPRegressor(max_iter=500, random_state=42, early_stopping=True)),
            ]),
            transformer=StandardScaler(),
        )

        grid_search = GridSearchCV(
            estimator=base_pipeline,
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
        """Executa 1 iteração de K-Fold Cross-Validation para a Rede Neural."""
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

        mlp_obj = self.__obter_mlp_estimator()
        camadas = str(getattr(mlp_obj, "hidden_layer_sizes", self.__params.get("hidden_layer_sizes", (100, 50))))
        ativacao = str(getattr(mlp_obj, "activation", self.__params.get("activation", "relu")))
        otimizador = str(getattr(mlp_obj, "solver", self.__params.get("solver", "adam")))
        alpha_val = float(getattr(mlp_obj, "alpha", self.__params.get("alpha", 0.01)))
        n_iter = int(getattr(mlp_obj, "n_iter_", 0))
        loss_val = float(getattr(mlp_obj, "loss_", 0.0))

        equacao_geral = self.obter_equacao_reta_geral()

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
            "camadas_ocultas": camadas,
            "ativacao": ativacao,
            "otimizador": otimizador,
            "alpha": round(alpha_val, 6),
            "epocas_executadas": n_iter,
            "loss_final": round(loss_val, 6),
            "equacao_geral": equacao_geral,
        }
