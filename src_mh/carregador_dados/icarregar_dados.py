from abc import ABC, abstractmethod
from typing import Generic, TypeVar

T = TypeVar("T")


class ICarregarDados(ABC, Generic[T]):

    @abstractmethod
    def carregar_dados(self) -> T:
        pass