import numpy as np
import pandas as pd
from typing import override
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import (
    mean_absolute_error,
    mean_squared_error,
    median_absolute_error,
    r2_score,
)
from sklearn.model_selection import validation_curve

from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo


class RegressaoLinearEstrategia(EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]):

    def __init__(self, fit_intercept: bool = True):
        self.__modelo = LinearRegression(fit_intercept=fit_intercept)
        self.__colunas: list[str] = []
        self.__param_range = np.logspace(-3, 2, 10)

    @property
    @override
    def nome(self) -> str:
        return "Regressão Linear"

    @override
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        self.__colunas = list(x_train.columns)
        self.__modelo.fit(x_train, y_train)

    @override
    def predizer(self, x: pd.DataFrame) -> np.ndarray:
        return self.__modelo.predict(x)

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
    def obter_curva_validacao(self, x: pd.DataFrame, y: pd.Series) -> dict[str, object]:
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
            "train_scores_mean": [round(float(v), 4) for v in np.mean(train_scores, axis=1)],
            "test_scores_mean": [round(float(v), 4) for v in np.mean(test_scores, axis=1)],
            "train_scores_std": [round(float(v), 4) for v in np.std(train_scores, axis=1)],
            "test_scores_std": [round(float(v), 4) for v in np.std(test_scores, axis=1)],
        }

    @override
    def obter_resultados(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, object]:
        y_pred = self.predizer(x_test)
        y_test_arr = np.asarray(y_test, dtype=float)
        y_pred_arr = np.asarray(y_pred, dtype=float)

        mae = mean_absolute_error(y_test_arr, y_pred_arr)
        mse = mean_squared_error(y_test_arr, y_pred_arr)
        rmse = np.sqrt(mse)
        medae = median_absolute_error(y_test_arr, y_pred_arr)
        r2 = r2_score(y_test_arr, y_pred_arr)

        smape = 100 * np.mean(2 * np.abs(y_pred_arr - y_test_arr) / (np.abs(y_test_arr) + np.abs(y_pred_arr) + 1e-8))
        bias = np.mean(y_pred_arr - y_test_arr)
        acc_10 = np.mean(np.abs(y_pred_arr - y_test_arr) / np.maximum(np.abs(y_test_arr), 1e-8) <= 0.10)

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

            # Curva de validação
            # "validation_curve": self.obter_curva_validacao(x_test, y_test),

            # Interpretabilidade
            "intercepto": round(float(self.__modelo.intercept_), 2),
            "coeficientes": coeficientes_dict,
            "interceptos_por_zona": self.obter_equacoes_por_zona(),
            "equacao_reta_geral": self.obter_equacao_reta_geral(),
        }

    @override
    def realizar_validacao_cruzada(self, x: pd.DataFrame, y: pd.Series, iteracao: int = 5) -> dict[str, object]:
        return {}
