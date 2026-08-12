from typing import Generic, Protocol, TypeVar

T_co = TypeVar("T_co", covariant=True)


class ICarregarDados(Protocol[T_co]):

    def carregar_dados(self) -> T_co:
        ...