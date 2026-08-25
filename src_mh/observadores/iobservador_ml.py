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
        self, nome_modelo: str, metricas: dict[str, object]
    ) -> None:
        """Notificado para registrar as métricas geradas por um modelo."""
        pass

    @abstractmethod
    def finalizar_experimento(self) -> None:
        """Notificado para finalizar e encerrar o ciclo do experimento."""
        pass
