from abc import ABC, abstractmethod
from typing import Generic, TypeVar

X_in = TypeVar("X_in")
Y_in = TypeVar("Y_in")
Y_out = TypeVar("Y_out")


class EstrategiaModelo(ABC, Generic[X_in, Y_in, Y_out]):

    @property
    @abstractmethod
    def nome(self) -> str:
        """Nome identificador do modelo."""
        pass

    @abstractmethod
    def treinar(self, x_train: X_in, y_train: Y_in) -> None:
        """Treina o modelo com os dados de treino fornecidos."""
        pass

    @abstractmethod
    def predizer(self, x: X_in) -> Y_out:
        """Realiza predições utilizando o modelo treinado."""
        pass

    @abstractmethod
    def obter_resultados(self, x_test: X_in, y_test: Y_in) -> dict[str, object]:
        """Avalia o modelo no conjunto de teste e retorna um dicionário de resultados."""
        pass

    def obter_curva_validacao(self, x: X_in, y: Y_in) -> dict[str, object]:
        """Calcula a curva de validação (opcional para modelos que suportam)."""
        return {}

    def realizar_validacao_cruzada(
        self,
        x: X_in,
        y: Y_in,
        iteracao: int = 5
    ) -> dict[str, object]:
        """Realiza validação cruzada (opcional para modelos que suportam)."""
        return {}

