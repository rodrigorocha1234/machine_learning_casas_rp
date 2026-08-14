from typing import Any, Protocol

import pandas as pd


class EstrategiaModelo(Protocol):

    @property
    def nome(self) -> str:
        ...

    def obter_curva_validacao(self, x: pd.DataFrame, y: pd.Series) -> dict[str, Any]:
        ...
    
    def obter_equacao_reta_geral(self) -> str:
        ...

    def treinar(self, x_train: pd.DataFrame, y_train: pd.Series) -> None:
        ...

    def predizer(self, x: pd.DataFrame) -> Any:
        ...

    def obter_resultados(self, x_test: pd.DataFrame, y_test: pd.Series) -> dict[str, Any]:
        ...
