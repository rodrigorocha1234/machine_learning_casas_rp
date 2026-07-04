from src.dados.ioperacao_dados import IoperacaoDados
from typing import Tuple, TypeVar, Generic
from abc import abstractmethod
import os


from typing import Generic, TypeVar

TArquivo = TypeVar("TArquivo")
TDados = TypeVar("TDados")


class Arquivo(IoperacaoDados[TDados], Generic[TArquivo, TDados]):
    def __init__(self,  nome_pasta_amarzenamento: str, nome_arquivo: str):
        self._caminho_base = os.getcwd()
        self._caminho_arquivo = os.path.join(
            self._caminho_base, nome_pasta_amarzenamento, nome_arquivo)

    @abstractmethod
    def salvar_dados(self, dados: TDados):
        ...

    @abstractmethod
    def atualizar_dados(self, dados: TDados):
        ...

    def verificar_arquivo(self) -> bool:
        """Método para verificar se o arquivo existe

        Returns:
            bool: verdadeiro se o arquivo existe, falso caso contrário
        """
        return os.path.exists(self._caminho_arquivo)