import logging
import os
import re
from typing import Any, override

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
from mlflow.models import infer_signature

from src_mh.observadores.iobservador_ml import IObservadorML

logger = logging.getLogger(__name__)


def _sanitizar_nome(nome: str) -> str:
    """Remove caracteres especiais de nomes para chaves de métricas e parâmetros do MLflow."""
    nome_sanitizado = re.sub(r"[^\w\s-]", "", nome).strip()
    return re.sub(r"[-\s]+", "_", nome_sanitizado)


def _extrair_metricas_num_planas(prefixo: str, dados: dict) -> dict[str, float]:
    """Filtra e extrai exclusivamente valores numéricos escalares para o mlflow.log_metrics."""
    metricas_planas: dict[str, float] = {}
    chaves_ignoradas = {
        "modelo_objeto", "x_sample", "y_sample", "figura_underfit",
        "figura_underfit_overfit", "scores_por_iteracao_rmse",
        "scores_por_iteracao_r2", "scores_por_iteracao_mae",
        "oof_scores_por_iteracao_rmse", "oof_scores_por_iteracao_r2",
        "oof_scores_por_iteracao_mae", "alpha_range", "param_range"
    }

    for chave, valor in dados.items():
        if chave in chaves_ignoradas or isinstance(valor, plt.Figure):
            continue

        chave_san = _sanitizar_nome(str(chave))
        nome_completo = f"{prefixo}_{chave_san}" if prefixo else chave_san

        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            metricas_planas[nome_completo] = float(valor)
        elif isinstance(valor, dict):
            metricas_planas.update(_extrair_metricas_num_planas(nome_completo, valor))

    return metricas_planas


class MLflowObservador(IObservadorML):
    """Observador Concreto que registra métricas, figuras e modelos no MLflow Server & Model Registry."""

    def __init__(
        self,
        tracking_uri: str | None = None,
        nome_experimento: str = "Previsão_Preço_Apartamentos_RP",
        nome_modelo_registry: str | None = None,
    ):
        self.__tracking_uri: str = (
            tracking_uri
            or os.getenv("MLFLOW_TRACKING_URI")
            or "http://localhost:5000"
        )
        self.__nome_experimento: str = nome_experimento
        self.__nome_modelo_registry: str | None = nome_modelo_registry

    @override
    def iniciar_experimento(
        self, nome_experimento: str, parametros: dict
    ) -> None:
        """Inicializa a conexão com o MLflow Server e inicia uma nova corrida (Run)."""
        try:
            mlflow.set_tracking_uri(self.__tracking_uri)

            try:
                exp = mlflow.get_experiment_by_name(self.__nome_experimento)
                if exp and exp.lifecycle_stage == "deleted":
                    self.__nome_experimento = f"{self.__nome_experimento}_Ativo"

                mlflow.set_experiment(self.__nome_experimento)
            except Exception:
                self.__nome_experimento = "Experimento_Previsão_Apartamentos_RP_Ativo"
                mlflow.set_experiment(self.__nome_experimento)

            nested = mlflow.active_run() is not None
            mlflow.start_run(run_name=nome_experimento, nested=nested)

            params_filtrados: dict[str, Any] = {
                _sanitizar_nome(str(k)): v
                for k, v in parametros.items()
                if isinstance(v, (int, float, str, bool))
            }
            if params_filtrados:
                mlflow.log_params(params_filtrados)

            logger.info(
                "Conectado ao MLflow Server (%s) - Run %siniciada: '%s'",
                self.__tracking_uri,
                "nested " if nested else "",
                nome_experimento,
            )
        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Conexão offline ou falha no MLflow Server: %s",
                e,
            )

    @override
    def registrar_metricas(
        self, metricas: dict
    ) -> None:
        """Ponto de entrada do observador para registrar todas as métricas, parâmetros e artefatos no MLflow."""
        try:
            run = mlflow.active_run()
            if run and getattr(run.info, "lifecycle_stage", "active") == "active":
                prefixo = _sanitizar_nome(str(metricas.get("nome_modelo", "")))

                # 1. Registra métricas numéricas escalares (Batch Log Metrics)
                self.__logar_metricas_num_planas(prefixo, metricas)

                # 2. Registra parâmetros, figuras e textos
                self.__logar_parametros_figuras_textos(prefixo, metricas)

                # 3. Registra o estimador/modelo no Model Registry se modelo_objeto estiver presente
                self.__registrar_modelo_no_registry(prefixo, metricas)

        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Aviso ao processar eventos no MLflow: %s",
                e,
            )

    def __logar_metricas_num_planas(
        self, prefixo: str, metricas: dict
    ) -> None:
        """1. Registra métricas numéricas escalares de forma otimizada em lote (Batching)."""
        metricas_num = _extrair_metricas_num_planas("", metricas)
        if metricas_num:
            step = (
                int(metricas["iteracao"])
                if "iteracao" in metricas and isinstance(metricas["iteracao"], (int, float))
                else None
            )
            mlflow.log_metrics(metricas_num, step=step)
            logger.info(
                "Métricas registradas no MLflow (%d métricas numéricas).",
                len(metricas_num),
            )

    def __logar_parametros_figuras_textos(
        self, prefixo: str, metricas: dict
    ) -> None:
        """2. Registra parâmetros textuais, equações como artefatos de texto e figuras Matplotlib."""
        for chave, valor in metricas.items():
            if chave in ("modelo_objeto", "x_sample", "y_sample"):
                continue

            chave_san = _sanitizar_nome(str(chave))

            if isinstance(valor, str):
                mlflow.log_param(chave_san, valor[:250])
                if "equacao" in chave.lower():
                    nome_txt = f"{chave_san}_{prefixo}.txt" if prefixo else f"{chave_san}.txt"
                    mlflow.log_text(valor, artifact_file=nome_txt)
                    logger.info("Equação registrada como artefato no MLflow: '%s'", nome_txt)

            elif isinstance(valor, bool):
                mlflow.log_param(chave_san, str(valor))

            elif isinstance(valor, dict) and ("intercepto" in chave.lower() or "coef" in chave.lower()):
                linhas_txt = [f"{k}: {v}" for k, v in valor.items()]
                conteudo_txt = "\n".join(linhas_txt)
                nome_txt = f"{chave_san}_{prefixo}.txt" if prefixo else f"{chave_san}.txt"
                mlflow.log_text(conteudo_txt, artifact_file=nome_txt)

            elif isinstance(valor, list) and valor:
                mlflow.log_param(f"vetor_{chave_san}", str(valor)[:250])
                if "scores_por_iteracao" in chave:
                    for idx, v in enumerate(valor):
                        if isinstance(v, (int, float)) and not isinstance(v, bool):
                            mlflow.log_metric(chave_san, float(v), step=idx + 1)

            elif isinstance(valor, plt.Figure):
                nome_fig = (
                    f"under_over_{prefixo}_{chave_san}.png"
                    if prefixo
                    else f"fig_{chave_san}.png"
                )
                mlflow.log_figure(valor, nome_fig)
                plt.close(valor)
                logger.info("Figura Matplotlib '%s' registrada no MLflow.", nome_fig)

    def __registrar_modelo_no_registry(
        self, prefixo: str, metricas: dict
    ) -> None:
        """3. Registra o estimador/modelo no Model Registry se modelo_objeto estiver presente."""
        modelo_obj = metricas.get("modelo_objeto")
        if modelo_obj is not None:
            try:
                x_sample = metricas.get("x_sample")
                y_sample = metricas.get("y_sample")
                signature = None

                if x_sample is not None and y_sample is not None:
                    try:
                        signature = infer_signature(x_sample, y_sample)
                    except Exception:
                        pass

                nome_registry = (
                    f"{self.__nome_modelo_registry}_{prefixo}"
                    if self.__nome_modelo_registry and prefixo
                    else (self.__nome_modelo_registry or "Modelo_Preco_Imoveis_RP")
                )

                mlflow.sklearn.log_model(
                    sk_model=modelo_obj,
                    name="modelo_scikit_learn",
                    signature=signature,
                    registered_model_name=nome_registry,
                )
                logger.info(
                    "Modelo registrado no Model Registry sob '%s'.", nome_registry
                )
            except Exception as ex_reg:
                logger.warning("⚠️ Aviso ao registrar modelo no MLflow: %s", ex_reg)

    @override
    def finalizar_experimento(self) -> None:
        """Finaliza a corrida ativa (Run) no MLflow Server se estiver ativa."""
        try:
            run = mlflow.active_run()
            if run and getattr(run.info, "lifecycle_stage", "active") == "active":
                mlflow.end_run()
                logger.info("Corrida do MLflow finalizada.")
        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Aviso ao finalizar corrida no MLflow: %s",
                e,
            )
