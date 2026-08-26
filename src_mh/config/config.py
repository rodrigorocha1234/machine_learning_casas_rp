import os
from pathlib import Path
from typing import Any, Final

import yaml

# Localização centralizada do arquivo de configuração YAML
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_YAML_PATH = CONFIG_DIR / "config.yaml"


def carregar_configuracao_yaml(caminho: Path | str | None = None) -> dict[str, Any]:
    """Carrega o arquivo YAML de configurações do projeto com suporte a fallbacks."""
    target_path = Path(caminho) if caminho else CONFIG_YAML_PATH

    if not target_path.exists():
        return {
            "modelo": {
                "treinamento_simples": {
                    "dados_separacao": {"dados_teste": 0.3, "random_state": 42},
                    "regressao_linear": {
                        "fit_intercept": True,
                        "param_range_start": -3,
                        "param_range_end": 2,
                        "param_range_num": 10,
                        "cv_folds": 5,
                        "scoring": "neg_root_mean_squared_error",
                        "n_jobs": 4,
                        "param_grid": {
                            "regressor__alpha": [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
                        },
                    },
                }
            },
            "mlflow": {
                "tracking_uri": os.getenv("MLFLOW_TRACKING_URI", "http://localhost:5000"),
                "nome_experimento": "Experimento_Previsão_Apartamentos_RP",
                "nome_modelo_registry": "Modelo_Preco_Imoveis_RP",
            },
            "carregador_dados": {
                "colunas": [
                    "Metragem",
                    "Quartos",
                    "Banheiros",
                    "Vagas",
                    "Bairro",
                    "Valor_da_Venda",
                ]
            },
        }

    with open(target_path, "r", encoding="utf-8") as f:
        dados: dict[str, Any] = yaml.safe_load(f) or {}
        return dados


_CONFIG_DATA: dict[str, Any] = carregar_configuracao_yaml()


class Config:
    """Classe estática para acesso às configurações do projeto."""
    _dados_sep = (
        _CONFIG_DATA.get("modelo", {})
        .get("treinamento_simples", {})
        .get("dados_separacao", {})
    )
    DADOS_TESTE: Final[float] = float(_dados_sep.get("dados_teste", 0.3))
    RANDOM_STATE: Final[int] = int(_dados_sep.get("random_state", 42))


def obter_config_regressao_linear() -> dict[str, Any]:
    """Retorna o dicionário de parâmetros para a estratégia de Regressão Linear lidos do YAML."""
    reg_cfg: dict[str, Any] = (
        _CONFIG_DATA.get("modelo", {})
        .get("treinamento_simples", {})
        .get("regressao_linear", {})
    )
    if not reg_cfg and "regressao_linear" in _CONFIG_DATA:
        reg_cfg = _CONFIG_DATA.get("regressao_linear", {})
    return reg_cfg


def obter_config_mlflow() -> dict[str, Any]:
    """Retorna o dicionário de parâmetros para a integração com o MLflow lidos do YAML."""
    mlflow_cfg: dict[str, Any] = _CONFIG_DATA.get("mlflow", {})
    return mlflow_cfg


def obter_config_carregador_dados() -> dict[str, Any]:
    """Retorna a lista de colunas e configurações para o carregador de dados lidos do YAML."""
    carregador_cfg: dict[str, Any] = _CONFIG_DATA.get("carregador_dados", {})
    return carregador_cfg