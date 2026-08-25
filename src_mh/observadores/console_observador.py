import logging
from typing import override

import matplotlib.pyplot as plt

from src_mh.observadores.iobservador_ml import IObservadorML

logger = logging.getLogger(__name__)


class ConsoleObservador(IObservadorML):

    @override
    def iniciar_experimento(
        self, nome_experimento: str, parametros: dict[str, object]
    ) -> None:
        logger.info("🚀 [CONSOLE OBSERVER] Experimento iniciado: '%s'", nome_experimento)
        logger.info("   Parâmetros da execução: %s", parametros)

    @override
    def registrar_metricas(
        self, nome_modelo: str, metricas: dict[str, object]
    ) -> None:
        logger.info("📊 [CONSOLE OBSERVER] Métricas registradas do modelo '%s':", nome_modelo)
        for k, v in metricas.items():
            if isinstance(v, (int, float, str)):
                logger.info("   - %s: %s", k, v)
            elif isinstance(v, plt.Figure):
                logger.info("   - %s: [Objeto Figure Matplotlib]", k)

    @override
    def finalizar_experimento(self) -> None:
        logger.info("🏁 [CONSOLE OBSERVER] Experimento concluído com sucesso.")
