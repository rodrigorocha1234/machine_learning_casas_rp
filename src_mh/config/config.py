import os
from pathlib import Path
from typing import Any, Final

import yaml

# Localização centralizada do arquivo de configuração YAML
CONFIG_DIR = Path(__file__).resolve().parent
CONFIG_YAML_PATH = CONFIG_DIR / "config.yaml"

class Config:
    with open(CONFIG_YAML_PATH, "r", encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)
        mlflow_config = config["mlflow"]
        tracking_uri = mlflow_config["tracking_uri"]
        nome_experimento = mlflow_config["nome_experimento"]
        nome_modelo_registry = mlflow_config["nome_modelo_registry"]

        treinamento_simples = config['treinamento_simples']['regressao_linear']

        fit_intercept_rl = treinamento_simples['fit_intercept']
        param_range_start_rl = treinamento_simples['param_range_start']
        param_range_end_rl = treinamento_simples['param_range_end']
        param_range_num_rl = treinamento_simples['param_range_num']
        cv_folds_rl = treinamento_simples['cv_folds']
        scoring_rl = treinamento_simples['scoring']
        n_jobs_rl = treinamento_simples['n_jobs']
        r_state_rl = treinamento_simples['random_state']



        turing_parametros_rl = config['turing_parametros']['regressao_linear']
        fit_intercept_turing_rl = turing_parametros_rl['fit_intercept']
        positive_turing_rl = turing_parametros_rl['positive']

        validacao_cruzada = config['validacao_cruzada']
        fit_intercept_rl_vl = validacao_cruzada['regressao_linear']['fit_intercept']
        positive_rl_vl = validacao_cruzada['regressao_linear']['positive']


