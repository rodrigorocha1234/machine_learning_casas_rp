from abc import abstractmethod, ABC
from typing import Any

import pandas as pd


class EstrategiaModelo(ABC):

    @property
    @abstractmethod
    def nome(self) -> str:
        ...

    @abstractmethod
    def obter_curva_validacao(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        ...

    @abstractmethod
    def obter_equacao_reta_geral(self) -> str:
        ...

    @abstractmethod
    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        ...

    @abstractmethod
    def predizer(self, x: pd.DataFrame) -> Any:
        ...

    @abstractmethod
    def obter_resultados(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        ...

    @abstractmethod
    def realizar_validacao_cruzada(
            self,
            x: pd.DataFrame,
            y: pd.Series,
            iteracao: int
    ) -> dict:
        ...

