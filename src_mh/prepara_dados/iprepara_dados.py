from abc import ABC, abstractmethod
from typing import TypeVar, Generic, Literal

T = TypeVar("T")


class IPrepararDados(ABC, Generic[T]):

    @abstractmethod
    def realizar_engenharia_atributos(self, df: T) -> T:
        pass

    @abstractmethod
    def separar_treino_teste(self, df_final: T,  tipo_escalonamento: Literal["standard", "minmax", None]) -> tuple[T, ...]:
        pass
