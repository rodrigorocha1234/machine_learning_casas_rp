from abc import ABC, abstractmethod
from typing import Any, Generic, TypeVar

from sklearn.model_selection import GridSearchCV

X_in = TypeVar("X_in")
Y_in = TypeVar("Y_in")
Y_out = TypeVar("Y_out")


class EstrategiaModelo(ABC, Generic[X_in, Y_in, Y_out]):

    @property
    @abstractmethod
    def nome(self) -> str:
        """Nome identificador do modelo."""
        pass

    @property
    def modelo_objeto(self) -> object | None:
        """Retorna o estimador/objeto de modelo treinado subjacente (ex: Sklearn, LightGBM, XGBoost)."""
        return None

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
        """Avalia o modelo no conjunto de teste e retorna um dicionário com métricas."""
        pass

    def obter_equacoes_por_zona(self) -> dict[str, float]:
        """Retorna interceptos por zona (opcional para modelos lineares)."""
        return {}

    def obter_equacao_reta_geral(self) -> str:
        """Retorna a equação geral da reta (opcional para modelos lineares)."""
        return ""

    def obter_curva_validacao(self, x: X_in, y: Y_in) -> dict[str, object]:
        """Calcula a curva de validação (opcional para modelos que suportam)."""
        return {}

    def gerar_figura_underfit_overfit(self, x: X_in, y: Y_in) -> Any:
        """Gera o objeto plt.Figure da curva de validação (opcional para modelos que suportam)."""
        return None

    def realizar_grid_search(
        self, x: X_in, y: Y_in
    ) -> GridSearchCV:
        """Realiza busca em grade de hiperparâmetros (GridSearchCV)."""
        return GridSearchCV(estimator=None, param_grid={})

    def obter_resultado_grid_search(
        self, grid_search: GridSearchCV
    ) -> dict[str, Any]:
        """Extrai e estrutura os resultados detalhados da busca em grade (GridSearchCV)."""
        return {}

    def realizar_validacao_cruzada(
        self, x: X_in, y: Y_in, iteracao: int = 5
    ) -> dict[str, Any]:
        """Realiza validação cruzada (opcional para modelos que suportam)."""
        return {}
