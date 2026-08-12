class FabricaModelo:

    @staticmethod
    def criar_modelo(nome_modelo: str) -> dict[str, str] :
        if nome_modelo == 'regressao_linear':
            return {
                'modelo': 'Linear',
                'avaliador': 'Avaliador'
            }
        return {}