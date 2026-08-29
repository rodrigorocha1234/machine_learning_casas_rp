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

        # SVR (Support Vector Regression) - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_svr = config.get('treinamento_simples', {}).get('regressao_svr', {})
        c_svr = float(treinamento_simples_svr.get('C', 100.0))
        epsilon_svr = float(treinamento_simples_svr.get('epsilon', 0.1))
        kernel_svr = str(treinamento_simples_svr.get('kernel', 'rbf'))
        gamma_svr = str(treinamento_simples_svr.get('gamma', 'scale'))

        turing_parametros_svr = config.get('turing_parametros', {}).get('regressao_svr', {})
        c_svr_turing = turing_parametros_svr.get('regressor__svr__C', [1.0, 10.0, 100.0, 1000.0])
        epsilon_svr_turing = turing_parametros_svr.get('regressor__svr__epsilon', [0.01, 0.1, 0.2])
        kernel_svr_turing = turing_parametros_svr.get('regressor__svr__kernel', ['rbf', 'linear'])

        validacao_cruzada_svr = config.get('validacao_cruzada', {}).get('regressao_svr', {})
        c_svr_vl = float(validacao_cruzada_svr.get('C', 100.0))
        epsilon_svr_vl = float(validacao_cruzada_svr.get('epsilon', 0.1))
        kernel_svr_vl = str(validacao_cruzada_svr.get('kernel', 'rbf'))

        # Rede Neural (MLPRegressor) - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_nn = config.get('treinamento_simples', {}).get('rede_neural', {})
        hidden_layer_sizes_nn = tuple(treinamento_simples_nn.get('hidden_layer_sizes', [100, 50]))
        activation_nn = str(treinamento_simples_nn.get('activation', 'relu'))
        solver_nn = str(treinamento_simples_nn.get('solver', 'adam'))
        alpha_nn = float(treinamento_simples_nn.get('alpha', 0.01))
        learning_rate_init_nn = float(treinamento_simples_nn.get('learning_rate_init', 0.01))
        max_iter_nn = int(treinamento_simples_nn.get('max_iter', 1000))
        early_stopping_nn = bool(treinamento_simples_nn.get('early_stopping', True))

        turing_parametros_nn = config.get('turing_parametros', {}).get('rede_neural', {})
        hidden_layer_sizes_nn_turing = turing_parametros_nn.get('regressor__mlp__hidden_layer_sizes', [[50], [100, 50], [100, 100]])
        activation_nn_turing = turing_parametros_nn.get('regressor__mlp__activation', ['relu', 'tanh'])
        alpha_nn_turing = turing_parametros_nn.get('regressor__mlp__alpha', [0.0001, 0.01, 1.0])

        validacao_cruzada_nn = config.get('validacao_cruzada', {}).get('rede_neural', {})
        hidden_layer_sizes_nn_vl = tuple(validacao_cruzada_nn.get('hidden_layer_sizes', [100, 50]))
        activation_nn_vl = str(validacao_cruzada_nn.get('activation', 'relu'))
        alpha_nn_vl = float(validacao_cruzada_nn.get('alpha', 0.01))
        max_iter_nn_vl = int(validacao_cruzada_nn.get('max_iter', 1000))

        # Gradient Boosting - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_gb = config.get('treinamento_simples', {}).get('gradient_boosting', {})
        n_estimators_gb = int(treinamento_simples_gb.get('n_estimators', 150))
        learning_rate_gb = float(treinamento_simples_gb.get('learning_rate', 0.1))
        max_depth_gb = int(treinamento_simples_gb.get('max_depth', 4))
        min_samples_split_gb = int(treinamento_simples_gb.get('min_samples_split', 5))
        min_samples_leaf_gb = int(treinamento_simples_gb.get('min_samples_leaf', 2))
        subsample_gb = float(treinamento_simples_gb.get('subsample', 0.9))

        turing_parametros_gb = config.get('turing_parametros', {}).get('gradient_boosting', {})
        n_estimators_gb_turing = turing_parametros_gb.get('n_estimators', [50, 100, 150, 200])
        learning_rate_gb_turing = turing_parametros_gb.get('learning_rate', [0.05, 0.1, 0.2])
        max_depth_gb_turing = turing_parametros_gb.get('max_depth', [3, 4, 5])

        validacao_cruzada_gb = config.get('validacao_cruzada', {}).get('gradient_boosting', {})
        n_estimators_gb_vl = int(validacao_cruzada_gb.get('n_estimators', 150))
        learning_rate_gb_vl = float(validacao_cruzada_gb.get('learning_rate', 0.1))
        max_depth_gb_vl = int(validacao_cruzada_gb.get('max_depth', 4))
        min_samples_split_gb_vl = int(validacao_cruzada_gb.get('min_samples_split', 5))
        min_samples_leaf_gb_vl = int(validacao_cruzada_gb.get('min_samples_leaf', 2))

        # XGBoost - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_xgb = config.get('treinamento_simples', {}).get('xgboost', {})
        n_estimators_xgb = int(treinamento_simples_xgb.get('n_estimators', 150))
        learning_rate_xgb = float(treinamento_simples_xgb.get('learning_rate', 0.08))
        max_depth_xgb = int(treinamento_simples_xgb.get('max_depth', 5))
        subsample_xgb = float(treinamento_simples_xgb.get('subsample', 0.85))
        colsample_bytree_xgb = float(treinamento_simples_xgb.get('colsample_bytree', 0.85))
        reg_alpha_xgb = float(treinamento_simples_xgb.get('reg_alpha', 0.1))
        reg_lambda_xgb = float(treinamento_simples_xgb.get('reg_lambda', 1.0))

        turing_parametros_xgb = config.get('turing_parametros', {}).get('xgboost', {})
        n_estimators_xgb_turing = turing_parametros_xgb.get('n_estimators', [50, 100, 150, 200])
        learning_rate_xgb_turing = turing_parametros_xgb.get('learning_rate', [0.03, 0.08, 0.15])
        max_depth_xgb_turing = turing_parametros_xgb.get('max_depth', [3, 5, 7])
        subsample_xgb_turing = turing_parametros_xgb.get('subsample', [0.7, 0.85, 1.0])

        validacao_cruzada_xgb = config.get('validacao_cruzada', {}).get('xgboost', {})
        n_estimators_xgb_vl = int(validacao_cruzada_xgb.get('n_estimators', 150))
        learning_rate_xgb_vl = float(validacao_cruzada_xgb.get('learning_rate', 0.08))
        max_depth_xgb_vl = int(validacao_cruzada_xgb.get('max_depth', 5))
        subsample_xgb_vl = float(validacao_cruzada_xgb.get('subsample', 0.85))
        colsample_bytree_xgb_vl = float(validacao_cruzada_xgb.get('colsample_bytree', 0.85))

        # LightGBM - Treinamento Simples, Tuning e Validação Cruzada
        treinamento_simples_lgb = config.get('treinamento_simples', {}).get('lightgbm', {})
        n_estimators_lgb = int(treinamento_simples_lgb.get('n_estimators', 150))
        learning_rate_lgb = float(treinamento_simples_lgb.get('learning_rate', 0.08))
        num_leaves_lgb = int(treinamento_simples_lgb.get('num_leaves', 31))
        max_depth_lgb = int(treinamento_simples_lgb.get('max_depth', -1))
        subsample_lgb = float(treinamento_simples_lgb.get('subsample', 0.85))
        colsample_bytree_lgb = float(treinamento_simples_lgb.get('colsample_bytree', 0.85))
        reg_alpha_lgb = float(treinamento_simples_lgb.get('reg_alpha', 0.1))
        reg_lambda_lgb = float(treinamento_simples_lgb.get('reg_lambda', 1.0))

        turing_parametros_lgb = config.get('turing_parametros', {}).get('lightgbm', {})
        n_estimators_lgb_turing = turing_parametros_lgb.get('n_estimators', [50, 100, 150, 200])
        learning_rate_lgb_turing = turing_parametros_lgb.get('learning_rate', [0.03, 0.08, 0.15])
        num_leaves_lgb_turing = turing_parametros_lgb.get('num_leaves', [15, 31, 63])
        subsample_lgb_turing = turing_parametros_lgb.get('subsample', [0.7, 0.85, 1.0])

        validacao_cruzada_lgb = config.get('validacao_cruzada', {}).get('lightgbm', {})
        n_estimators_lgb_vl = int(validacao_cruzada_lgb.get('n_estimators', 150))
        learning_rate_lgb_vl = float(validacao_cruzada_lgb.get('learning_rate', 0.08))
        num_leaves_lgb_vl = int(validacao_cruzada_lgb.get('num_leaves', 31))
        subsample_lgb_vl = float(validacao_cruzada_lgb.get('subsample', 0.85))
        colsample_bytree_lgb_vl = float(validacao_cruzada_lgb.get('colsample_bytree', 0.85))
