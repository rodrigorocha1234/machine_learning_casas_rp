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

# Precompilação do Regex para máxima performance na sanitização de nomes
_REGEX_CARACTERES_INVALIDOS = re.compile(r"[^a-zA-Z0-9_\-\.\:\/ ]")


def _sanitizar_nome(nome: str) -> str:
    """Substitui caracteres inválidos por '_' para atender às exigências de nomes do MLflow."""
    return _REGEX_CARACTERES_INVALIDOS.sub("_", nome)


def _extrair_metricas_planas(
    prefixo: str, dados: dict[str, Any]
) -> dict[str, float]:
    """Extrai recursivamente métricas numéricas em um dicionário plano para envio em lote (batch)."""
    metricas_planas: dict[str, float] = {}
    chaves_ignoradas = {"modelo_objeto", "x_sample", "y_sample"}

    for chave, valor in dados.items():
        # Ignora objetos de figura, amostragem ou modelo na extração de métricas numéricas
        if isinstance(valor, plt.Figure) or chave in chaves_ignoradas:
            continue

        chave_san = _sanitizar_nome(str(chave))
        nome_completo = f"{prefixo}_{chave_san}" if prefixo else chave_san

        if isinstance(valor, (int, float)) and not isinstance(valor, bool):
            metricas_planas[nome_completo] = float(valor)
        elif isinstance(valor, dict):
            metricas_planas.update(_extrair_metricas_planas(nome_completo, valor))

    return metricas_planas


class MLflowObservador(IObservadorML):
    """Observador Concreto que escuta eventos do Pipeline e registra métricas, figuras, equações e modelos no MLflow Server & Model Registry."""

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
        self, nome_experimento: str, parametros: dict[str, object]
    ) -> None:
        """Inicializa a conexão com o MLflow Server e inicia uma nova corrida (Run)."""
        try:
            mlflow.set_tracking_uri(self.__tracking_uri)

            exp = mlflow.get_experiment_by_name(self.__nome_experimento)
            if exp and exp.lifecycle_stage == "deleted":
                self.__nome_experimento = f"{self.__nome_experimento}_V2"

            mlflow.set_experiment(self.__nome_experimento)
            mlflow.start_run(run_name=nome_experimento)

            params_filtrados: dict[str, Any] = {
                _sanitizar_nome(k): v
                for k, v in parametros.items()
                if isinstance(v, (int, float, str, bool))
            }
            if params_filtrados:
                mlflow.log_params(params_filtrados)

            logger.info(
                "Conectado ao MLflow Server (%s) - Run iniciada: '%s'",
                self.__tracking_uri,
                nome_experimento,
            )
        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Conexão offline ou falha no MLflow Server: %s",
                e,
            )

    @override
    def registrar_metricas(
        self, nome_modelo: str, metricas: dict[str, object]
    ) -> None:
        """Ponto de entrada do observador para registrar todas as métricas e artefatos de um modelo."""
        try:
            if mlflow.active_run():
                prefixo = _sanitizar_nome(nome_modelo)

                # 1. Registra parâmetros da busca em grade (hiperparâmetros otimizados)
                self.__logar_parametros_tuning(prefixo, metricas, nome_modelo)

                # 2. Registra métricas numéricas em lote (Batching)
                self.__logar_metricas_em_lote(prefixo, metricas, nome_modelo)

                # 3. Registra gráficos e figuras Matplotlib como artefatos de imagem
                self.__logar_figuras_matplotlib(prefixo, metricas, nome_modelo)

                # 4. Registra equações matemáticas e parâmetros textuais
                self.__logar_equacoes_e_textos(prefixo, metricas, nome_modelo)

                # 5. Registra o estimador treinado com Assinatura (Schema Inputs & Outputs) no Model Registry
                self.__registrar_modelo_no_registry(prefixo, metricas, nome_modelo)

        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Aviso ao processar eventos no MLflow: %s",
                e,
            )

    def __logar_parametros_tuning(
        self, prefixo: str, metricas: dict[str, object], nome_modelo: str
    ) -> None:
        """Registra parâmetros do tuning (best_params_) no MLflow."""
        melhores_params = metricas.get("melhores_parametros")
        if isinstance(melhores_params, dict):
            params_sanitizados = {
                f"{prefixo}_{_sanitizar_nome(str(k))}": str(v)
                for k, v in melhores_params.items()
            }
            if params_sanitizados:
                mlflow.log_params(params_sanitizados)
                logger.info(
                    "Parâmetros de tuning do modelo '%s' (%d params) registrados no MLflow.",
                    nome_modelo,
                    len(params_sanitizados),
                )

    def __logar_metricas_em_lote(
        self, prefixo: str, metricas: dict[str, object], nome_modelo: str
    ) -> None:
        """Extrai e registra todas as métricas numéricas de forma otimizada em uma única requisição HTTP (batch)."""
        metricas_batch = _extrair_metricas_planas(prefixo, metricas)
        if metricas_batch:
            mlflow.log_metrics(metricas_batch)
            logger.info(
                "Métricas do modelo '%s' (%d métricas) registradas em lote (batch) no MLflow.",
                nome_modelo,
                len(metricas_batch),
            )

    def __logar_figuras_matplotlib(
        self, prefixo: str, metricas: dict[str, object], nome_modelo: str
    ) -> None:
        """Filtra objetos plt.Figure no dicionário de métricas e os salva como artefatos PNG no MLflow."""
        for chave, valor in metricas.items():
            if isinstance(valor, plt.Figure):
                nome_fig_artifact = (
                    f"under_over_{prefixo}_{_sanitizar_nome(chave)}.png"
                )
                mlflow.log_figure(valor, nome_fig_artifact)
                plt.close(valor)
                logger.info(
                    "Figura Matplotlib '%s' vinda do modelo '%s' registrada com sucesso no MLflow.",
                    nome_fig_artifact,
                    nome_modelo,
                )

    def __logar_equacoes_e_textos(
        self, prefixo: str, metricas: dict[str, object], nome_modelo: str
    ) -> None:
        """Registra parâmetros do tipo texto (ex: equações matemáticas da reta) como parâmetros e arquivos .txt no MLflow."""
        for chave, valor in metricas.items():
            if isinstance(valor, str):
                chave_san = _sanitizar_nome(chave)
                mlflow.log_param(f"{prefixo}_{chave_san}", valor)

                if "equacao" in chave.lower():
                    nome_txt_artifact = f"equacao_reta_{prefixo}.txt"
                    mlflow.log_text(valor, nome_txt_artifact)
                    logger.info(
                        "Equação da reta do modelo '%s' registrada como parâmetro e artefato de texto ('%s') no MLflow.",
                        nome_modelo,
                        nome_txt_artifact,
                    )

    def __registrar_modelo_no_registry(
        self, prefixo: str, metricas: dict[str, object], nome_modelo: str
    ) -> None:
        """Serializa o estimador treinado com assinatura de Schema (Inputs/Outputs) e publica no Model Registry."""
        modelo_obj = metricas.get("modelo_objeto")
        if modelo_obj is not None:
            try:
                nome_registry = (
                    self.__nome_modelo_registry or f"Modelo_{prefixo}"
                )

                # Inferência do Esquema de Entradas (Inputs) e Saídas (Outputs)
                x_sample = metricas.get("x_sample")
                y_sample = metricas.get("y_sample")
                signature = None

                if x_sample is not None and y_sample is not None:
                    try:
                        signature = infer_signature(x_sample, y_sample)
                    except Exception as ex_sig:
                        logger.warning(
                            "⚠️ Aviso ao inferir assinatura do esquema MLflow: %s",
                            ex_sig,
                        )

                mlflow.sklearn.log_model(
                    sk_model=modelo_obj,
                    artifact_path="modelo_scikit_learn",
                    signature=signature,
                    registered_model_name=nome_registry,
                )
                logger.info(
                    "Modelo '%s' registrado no MLflow Model Registry sob o nome '%s' (com esquema de Inputs/Outputs).",
                    nome_modelo,
                    nome_registry,
                )
            except Exception as ex_reg:
                logger.warning(
                    "⚠️ Aviso ao registrar modelo no Model Registry: %s", ex_reg
                )

    @override
    def finalizar_experimento(self) -> None:
        """Finaliza a corrida ativa (Run) no MLflow Server."""
        try:
            if mlflow.active_run():
                mlflow.end_run()
                logger.info("Corrida do MLflow finalizada.")
        except Exception as e:
            logger.warning(
                "⚠️ [MLflowObservador] Aviso ao finalizar corrida no MLflow: %s",
                e,
            )
