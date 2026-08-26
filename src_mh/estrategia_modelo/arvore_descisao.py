import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeRegressor

from src_mh.estrategia_modelo.estrategia_modelo import EstrategiaModelo, X_in, Y_in, Y_out


class ArvoreDescisaoEstrategia(
    EstrategiaModelo[pd.DataFrame, pd.Series, np.ndarray]
):

    def __init__(self):



    @property
    def nome(self) -> str:
        pass

    def treinar(self, x_train: X_in, y_train: Y_in) -> None:
        pass

    def predizer(self, x: X_in) -> Y_out:
        pass

    def obter_resultados(self, x_test: X_in, y_test: Y_in) -> dict[str, object]:
        pass