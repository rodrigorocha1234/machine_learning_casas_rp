import os
from typing import Final

import yaml


class Config:
    with open(os.path.join(os.getcwd(), 'src_mh', 'config', 'config.yaml'), "r", encoding="utf-8") as arquivo:
        config = yaml.safe_load(arquivo)
        DADOS_TESTE : Final[float] = config['modelo']['treinamento_simples']['dados_separacao']['dados_teste']
        RANDOM_STATE : Final[float] = config['modelo']['treinamento_simples']['dados_separacao']['random_state']