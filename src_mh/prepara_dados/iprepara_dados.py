from typing import Literal, Protocol, TypeVar

T = TypeVar("T")
X_co = TypeVar("X_co", covariant=True)
Y_co = TypeVar("Y_co", covariant=True)


class IPrepararDados(Protocol[T, X_co, Y_co]):

    def realizar_engenharia_atributos(
        self,
        df: T
    ) -> T:
        ...

    def separar_treino_teste(
        self,
        df_final: T,
        tipo_escalonamento: Literal["standard", "minmax", None]
    ) -> tuple[X_co, X_co, Y_co, Y_co]:
        ...