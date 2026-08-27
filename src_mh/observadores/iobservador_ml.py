from abc import ABC, abstractmethod


class IObservadorML(ABC):

    @abstractmethod
    def iniciar_experimento(
        self, nome_experimento: str, parametros: dict[str, object]
    ) -> None:
        """Notificado para iniciar a execução de um experimento de ML."""
        pass

    @abstractmethod
    def registrar_metricas(
            self, metricas: dict
    ) -> None:
        """Ponto de entrada do observador para registrar todas as métricas e  de um modelo."""

    @abstractmethod
    def finalizar_experimento(self) -> None:
        """Notificado para finalizar e encerrar o ciclo do experimento."""
        pass
