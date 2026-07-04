from abc import ABC, abstractmethod
from typing import Generic, TypeVar

TDados = TypeVar("TDados")


class IoperacaoDados(ABC, Generic[TDados]):

    @abstractmethod
    def salvar_dados(self, dados: TDados):
        ...

    @abstractmethod
    def atualizar_dados(self, dados: TDados):
        ...