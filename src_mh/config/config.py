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

        # Regressão Linear - Tuning e Validação Cruzada
        turing_parametros_rl = config.get('turing_parametros', {}).get('regressao_linear', {})
        fit_intercept_turing_rl = turing_parametros_rl.get('fit_intercept', [True, False])
        positive_turing_rl = turing_parametros_rl.get('positive', [True, False])

        validacao_cruzada_rl = config.get('validacao_cruzada', {}).get('regressao_linear', {})
        fit_intercept_rl_vl = validacao_cruzada_rl.get('fit_intercept', True)
        positive_rl_vl = validacao_cruzada_rl.get('positive', True)

        # Regressão Ridge - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_ridge = config.get('treinamento_simples', {}).get('regressao_ridge', {})
        alpha_ridge = treinamento_simples_ridge.get('alpha', 1.0)
        fit_intercept_ridge = treinamento_simples_ridge.get('fit_intercept', True)
        solver_ridge = treinamento_simples_ridge.get('solver', 'auto')

        turing_parametros_ridge = config.get('turing_parametros', {}).get('regressao_ridge', {})
        alpha_ridge_turing = turing_parametros_ridge.get('alpha', [0.01, 0.1, 1.0, 10.0, 100.0])
        fit_intercept_turing_ridge = turing_parametros_ridge.get('fit_intercept', [True, False])
        solver_ridge_turing = turing_parametros_ridge.get('solver', ['auto', 'svd', 'cholesky', 'lsqr'])

        validacao_cruzada_ridge = config.get('validacao_cruzada', {}).get('regressao_ridge', {})
        alpha_ridge_vl = validacao_cruzada_ridge.get('alpha', 1.0)
        fit_intercept_ridge_vl = validacao_cruzada_ridge.get('fit_intercept', True)

        # Regressão Lasso - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_lasso = config.get('treinamento_simples', {}).get('regressao_lasso', {})
        alpha_lasso = treinamento_simples_lasso.get('alpha', 1.0)
        fit_intercept_lasso = treinamento_simples_lasso.get('fit_intercept', True)
        max_iter_lasso = treinamento_simples_lasso.get('max_iter', 15000)
        tol_lasso = treinamento_simples_lasso.get('tol', 0.001)

        turing_parametros_lasso = config.get('turing_parametros', {}).get('regressao_lasso', {})
        alpha_lasso_turing = turing_parametros_lasso.get('alpha', [0.01, 0.1, 1.0, 10.0, 100.0])
        fit_intercept_turing_lasso = turing_parametros_lasso.get('fit_intercept', [True, False])
        selection_turing_lasso = turing_parametros_lasso.get('selection', ['cyclic', 'random'])

        validacao_cruzada_lasso = config.get('validacao_cruzada', {}).get('regressao_lasso', {})
        alpha_lasso_vl = validacao_cruzada_lasso.get('alpha', 1.0)
        fit_intercept_lasso_vl = validacao_cruzada_lasso.get('fit_intercept', True)

        # Regressão ElasticNet - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_en = config.get('treinamento_simples', {}).get('regressao_elasticnet', {})
        alpha_elasticnet = treinamento_simples_en.get('alpha', 1.0)
        l1_ratio_elasticnet = treinamento_simples_en.get('l1_ratio', 0.5)
        fit_intercept_elasticnet = treinamento_simples_en.get('fit_intercept', True)
        max_iter_elasticnet = treinamento_simples_en.get('max_iter', 15000)
        tol_elasticnet = treinamento_simples_en.get('tol', 0.001)

        turing_parametros_en = config.get('turing_parametros', {}).get('regressao_elasticnet', {})
        alpha_elasticnet_turing = turing_parametros_en.get('alpha', [0.01, 0.1, 1.0, 10.0, 100.0])
        l1_ratio_elasticnet_turing = turing_parametros_en.get('l1_ratio', [0.1, 0.3, 0.5, 0.7, 0.9])
        fit_intercept_turing_elasticnet = turing_parametros_en.get('fit_intercept', [True, False])

        validacao_cruzada_en = config.get('validacao_cruzada', {}).get('regressao_elasticnet', {})
        alpha_elasticnet_vl = validacao_cruzada_en.get('alpha', 1.0)
        l1_ratio_elasticnet_vl = validacao_cruzada_en.get('l1_ratio', 0.5)
        fit_intercept_elasticnet_vl = validacao_cruzada_en.get('fit_intercept', True)

        # Árvore de Decisão - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_dt = config.get('treinamento_simples', {}).get('arvore_decisao', {})
        max_depth_dt = treinamento_simples_dt.get('max_depth', 6)
        min_samples_split_dt = treinamento_simples_dt.get('min_samples_split', 5)
        min_samples_leaf_dt = treinamento_simples_dt.get('min_samples_leaf', 2)
        r_state_dt = treinamento_simples_dt.get('random_state', 42)

        turing_parametros_dt = config.get('turing_parametros', {}).get('arvore_decisao', {})
        max_depth_turing_dt = turing_parametros_dt.get('max_depth', [3, 5, 7, 10, None])
        min_samples_split_turing_dt = turing_parametros_dt.get('min_samples_split', [2, 5, 10])
        min_samples_leaf_turing_dt = turing_parametros_dt.get('min_samples_leaf', [1, 2, 4])

        validacao_cruzada_dt = config.get('validacao_cruzada', {}).get('arvore_decisao', {})
        max_depth_dt_vl = validacao_cruzada_dt.get('max_depth', 6)
        min_samples_split_dt_vl = validacao_cruzada_dt.get('min_samples_split', 5)
        min_samples_leaf_dt_vl = validacao_cruzada_dt.get('min_samples_leaf', 2)

        # Regressão Polinomial - Treinamento Simples
        treinamento_simples_rp = config.get('treinamento_simples', {}).get('regressao_polinomial', {})
        degree_rp = treinamento_simples_rp.get('degree', 2)
        include_bias_rp = treinamento_simples_rp.get('include_bias', False)
        fit_intercept_rp = treinamento_simples_rp.get('fit_intercept', True)
        positive_rp = treinamento_simples_rp.get('positive', False)

        # Regressão Polinomial - Tuning de Parâmetros
        turing_parametros_rp = config.get('turing_parametros', {}).get('regressao_polinomial', {})
        poly_degree_turing_rp = turing_parametros_rp.get('poly__degree', [1, 2, 3])
        fit_intercept_turing_rp = turing_parametros_rp.get('regressor__fit_intercept', [True, False])
        positive_turing_rp = turing_parametros_rp.get('regressor__positive', [True, False])

        # Regressão Polinomial - Validação Cruzada
        validacao_cruzada_rp = config.get('validacao_cruzada', {}).get('regressao_polinomial', {})
        degree_rp_vl = validacao_cruzada_rp.get('degree', 2)
        include_bias_rp_vl = validacao_cruzada_rp.get('include_bias', False)
        fit_intercept_rp_vl = validacao_cruzada_rp.get('fit_intercept', True)
        positive_rp_vl = validacao_cruzada_rp.get('positive', False)

        # Random Forest - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_rf = config.get('treinamento_simples', {}).get('random_forest', {})
        n_estimators_rf = treinamento_simples_rf.get('n_estimators', 100)
        max_depth_rf = treinamento_simples_rf.get('max_depth', 10)
        min_samples_split_rf = treinamento_simples_rf.get('min_samples_split', 5)
        min_samples_leaf_rf = treinamento_simples_rf.get('min_samples_leaf', 2)

        turing_parametros_rf = config.get('turing_parametros', {}).get('random_forest', {})
        n_estimators_rf_turing = turing_parametros_rf.get('n_estimators', [50, 100, 200])
        max_depth_rf_turing = turing_parametros_rf.get('max_depth', [5, 10, 15, None])
        min_samples_split_rf_turing = turing_parametros_rf.get('min_samples_split', [2, 5, 10])
        min_samples_leaf_rf_turing = turing_parametros_rf.get('min_samples_leaf', [1, 2, 4])

        validacao_cruzada_rf = config.get('validacao_cruzada', {}).get('random_forest', {})
        n_estimators_rf_vl = validacao_cruzada_rf.get('n_estimators', 100)
        max_depth_rf_vl = validacao_cruzada_rf.get('max_depth', 10)
        min_samples_split_rf_vl = validacao_cruzada_rf.get('min_samples_split', 5)
        min_samples_leaf_rf_vl = validacao_cruzada_rf.get('min_samples_leaf', 2)
