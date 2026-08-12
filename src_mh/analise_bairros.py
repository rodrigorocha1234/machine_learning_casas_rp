import pandas as pd


def classificar_zonas(df: pd.DataFrame) -> pd.DataFrame:
    """
    Classifica os bairros em zonas da cidade.
    Cria uma nova coluna 'Zona' baseada na coluna 'Bairro'.
    """
    df_copy = df.copy()

    if 'Bairro' not in df_copy.columns:
        raise ValueError("O DataFrame deve conter a coluna 'Bairro'.")

    # Dicionário de mapeamento exaustivo para todos os 153 bairros do dataset
    mapa_zonas = {'Adao do Carmo Leonel': 'Zona Norte', 'Adelino Simioni': 'Zona Norte',
        'Alamedas do Botanico': 'Zona Sul', 'Alto da Boa Vista': 'Zona Oeste', 'Alto do Ipiranga': 'Zona Norte',
        'Bonfim Paulista': 'Zona Sul', 'Bosque das Juritis': 'Zona Sul', 'Campos Eliseos': 'Zona Norte',
        'Castelo Branco': 'Zona Leste', 'Centro': 'Centro', 'Chacaras Bonacorsi': 'Zona Leste',
        'Chacaras Pedro Correa de Carvalho': 'Zona Leste', 'City Ribeirao': 'Zona Leste',
        'Condominio Itamaraty': 'Zona Sul', 'Condominio Mirante Sul': 'Zona Sul', 'Condominio Uirapuru': 'Zona Sul',
        'Condominio Vila Florenca': 'Zona Sul', 'Conjunto Habitacional Jardim das Palmeiras': 'Zona Leste',
        'Conjunto Habitacional Silvio Passalacqua': 'Zona Norte', 'Dom Bernardo Jose Mielle': 'Zona Norte',
        'Geraldo Correia de Carvalho': 'Zona Norte', 'Higienopolis': 'Centro', 'Iguatemi': 'Zona Leste',
        'Independencia': 'Zona Leste', 'Ipiranga': 'Zona Norte', 'Jamil Seme Cury': 'Zona Norte',
        'Jardim America': 'Zona Oeste', 'Jardim Angelo Jurca': 'Zona Norte', 'Jardim Anhanguera': 'Zona Leste',
        'Jardim Antartica': 'Zona Oeste', 'Jardim Bela Vista': 'Zona Oeste', 'Jardim Bom Pastor': 'Zona Leste',
        'Jardim Botanico': 'Zona Sul', 'Jardim California': 'Zona Sul', 'Jardim Canada': 'Zona Sul',
        'Jardim Castelo Branco': 'Zona Leste', 'Jardim Centenario': 'Zona Oeste', 'Jardim Cybelli': 'Zona Norte',
        'Jardim Diva Tarla de Carvalho': 'Zona Norte', 'Jardim Doutor Paulo Gomes Romeo': 'Zona Oeste',
        'Jardim Eugenio Mendes Lopes': 'Zona Norte', 'Jardim Florestan Fernandes': 'Zona Norte',
        'Jardim Florida': 'Zona Sul', 'Jardim Formoso': 'Zona Oeste', 'Jardim Guapore': 'Zona Sul',
        'Jardim Heitor Rigon': 'Zona Norte', 'Jardim Helena': 'Zona Leste', 'Jardim Herculano Fernandes': 'Zona Norte',
        'Jardim Ilhas do Sul': 'Zona Sul', 'Jardim Interlagos': 'Zona Leste', 'Jardim Iraja': 'Zona Sul',
        'Jardim Itapora': 'Zona Norte', 'Jardim Itau': 'Zona Oeste', 'Jardim Jandaia': 'Zona Norte',
        'Jardim Javari': 'Zona Norte', 'Jardim Joao Rossi': 'Zona Sul', 'Jardim Jose Figueira': 'Zona Sul',
        'Jardim Jose Sampaio Junior': 'Zona Norte', 'Jardim Jose Wilson Toni': 'Zona Norte',
        'Jardim Macedo': 'Zona Leste', 'Jardim Manoel Penna': 'Zona Leste', 'Jardim Marchesi': 'Zona Oeste',
        'Jardim Maria Goretti': 'Zona Oeste', 'Jardim Mosteiro': 'Zona Norte', 'Jardim Nova Alianca Sul': 'Zona Sul',
        'Jardim Novo Mundo': 'Zona Leste', 'Jardim Olhos DAgua': 'Zona Sul', 'Jardim Ouro Branco': 'Zona Leste',
        'Jardim Paiva': 'Zona Oeste', 'Jardim Palma Travassos': 'Zona Leste', 'Jardim Paulista': 'Zona Leste',
        'Jardim Paulistano': 'Zona Leste', 'Jardim Pedra Branca': 'Zona Leste', 'Jardim Presidente Dutra': 'Zona Norte',
        'Jardim Presidente Dutra II': 'Zona Norte', 'Jardim Recreio': 'Zona Oeste', 'Jardim Regatas': 'Zona Norte',
        'Jardim Saint Gerard': 'Zona Sul', 'Jardim San Marco': 'Zona Sul', 'Jardim Santa Cecilia': 'Zona Sul',
        'Jardim Sao Fernando': 'Zona Leste', 'Jardim Sao Jose': 'Zona Leste', 'Jardim Sao Luiz': 'Zona Sul',
        'Jardim Sumare': 'Zona Sul', 'Jardim Vilico Cantarelli': 'Zona Oeste', 'Jardim Villarica': 'Zona Sul',
        'Jardim Zaneti': 'Zona Sul', 'Jardim Zara': 'Zona Leste', 'Jardim das Oliveiras': 'Zona Leste',
        'Jardim das Palmeiras': 'Zona Leste', 'Jardim do Trevo': 'Zona Leste', 'Lagoinha': 'Zona Leste',
        'Loteamento Santa Marta': 'Zona Sul', 'Nova Alianca': 'Zona Sul', 'Nova Ribeirania': 'Zona Leste',
        'Panamby II': 'Zona Sul', 'Parque Anhanguera': 'Zona Leste', 'Parque Industrial Lagoinha': 'Zona Leste',
        'Parque Residencial Candido Portinari': 'Zona Leste', 'Parque Residencial Lagoinha': 'Zona Leste',
        'Parque Ribeirao Preto': 'Zona Oeste', 'Parque Sao Sebastiao': 'Zona Leste',
        'Parque das Oliveiras': 'Zona Norte', 'Parque das Oliveiras II': 'Zona Norte',
        'Parque dos Bandeirantes': 'Zona Leste', 'Parque dos Lagos': 'Zona Leste', 'Parque dos Pinus': 'Zona Norte',
        'Planalto Verde': 'Zona Oeste', 'Plazas de Espana': 'Zona Sul', 'Presidente Medici': 'Zona Leste',
        'Quinta da Primavera': 'Zona Sul', 'Quintas de Sao Jose': 'Zona Sul', 'Quintino Facci II': 'Zona Norte',
        'Real Sul': 'Zona Sul', 'Recanto das Palmeiras': 'Zona Leste', 'Recreio Anhanguera': 'Zona Leste',
        'Recreio das Acacias': 'Zona Leste', 'Republica': 'Zona Oeste', 'Reserva Macauba': 'Zona Norte',
        'Reserva Real': 'Zona Sul', 'Reserva Sao Jose': 'Zona Leste', 'Reserva Sul Condominio': 'Zona Sul',
        'Residencial Alto do Ipe': 'Zona Sul', 'Residencial Florida': 'Zona Sul',
        'Residencial Greenville': 'Zona Leste', 'Residencial Jequitiba': 'Zona Sul',
        'Residencial Monterrey': 'Zona Leste', 'Residencial Morro do Ipe': 'Zona Sul',
        'Residencial Parque dos Servidores': 'Zona Leste', 'Residencial Taiwan': 'Zona Sul',
        'Residencial das Americas': 'Zona Leste', 'Residencial e Comercial Palmares': 'Zona Leste',
        'Ribeirania': 'Zona Leste', 'Ribeirao Verde': 'Zona Norte', 'Santa Cruz do Jose Jacques': 'Zona Sul',
        'Setor Central': 'Centro', 'Sumarezinho': 'Zona Oeste', 'Valentina Figueiredo': 'Zona Norte',
        'Vila Abranches': 'Zona Leste', 'Vila Albertina': 'Zona Norte', 'Vila Amelia': 'Zona Oeste',
        'Vila Ana Maria': 'Zona Sul', 'Vila Elisa': 'Zona Norte', 'Vila Guiomar': 'Zona Oeste',
        'Vila Maria Luiza': 'Zona Sul', 'Vila Mariana': 'Zona Norte', 'Vila Monte Alegre': 'Zona Oeste',
        'Vila Recreio': 'Zona Norte', 'Vila Seixas': 'Centro', 'Vila Tamandare': 'Zona Norte',
        'Vila Tiberio': 'Zona Oeste', 'Vila Virginia': 'Zona Oeste', 'Vila do Golf': 'Zona Sul'}

    df_copy['Zona'] = df_copy['Bairro'].map(mapa_zonas).fillna('Centro/Outros')
    return df_copy


def calcular_media_bairro(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a média de valor de venda (Valor_da_Venda) por Bairro.
    """
    if 'Valor_da_Venda' not in df.columns or 'Bairro' not in df.columns:
        raise ValueError("O DataFrame deve conter as colunas 'Valor_da_Venda' e 'Bairro'.")

    media_bairro = df.groupby('Bairro')['Valor_da_Venda'].mean().reset_index()
    media_bairro = media_bairro.sort_values(by='Valor_da_Venda', ascending=False)
    return media_bairro


def calcular_media_zona(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula a média de valor de venda (Valor_da_Venda) por Zona.
    Se a coluna 'Zona' não existir, a função de classificação será chamada.
    """
    if 'Valor_da_Venda' not in df.columns:
        raise ValueError("O DataFrame deve conter a coluna 'Valor_da_Venda'.")

    df_zonas = df.copy()
    if 'Zona' not in df_zonas.columns:
        if 'Bairro' in df_zonas.columns:
            df_zonas = classificar_zonas(df_zonas)
        else:
            raise ValueError("O DataFrame deve conter a coluna 'Bairro' para classificar as Zonas e calcular a média.")

    media_zona = df_zonas.groupby('Zona')['Valor_da_Venda'].mean().reset_index()
    media_zona = media_zona.sort_values(by='Valor_da_Venda', ascending=False)
    return media_zona


def calcular_media_m2_zona(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calcula o valor do metro quadrado para cada imóvel e
    retorna a média desse valor (valor_m2) por Zona.
    """
    if 'Valor_da_Venda' not in df.columns or 'Metragem' not in df.columns:
        raise ValueError("O DataFrame deve conter as colunas 'Valor_da_Venda' e 'Metragem'.")

    df_m2 = df.copy()

    # Calcular o valor_m2, tratando possíveis divisões por zero ou nulos
    # Caso Metragem seja 0, pode gerar inf, então é ideal tratar se houver
    df_m2['valor_m2'] = df_m2['Valor_da_Venda'] / df_m2['Metragem']

    if 'Zona' not in df_m2.columns:
        if 'Bairro' in df_m2.columns:
            df_m2 = classificar_zonas(df_m2)
        else:
            raise ValueError("O DataFrame deve conter a coluna 'Bairro' para classificar as Zonas.")

    media_m2_zona = df_m2.groupby('Zona')['valor_m2'].mean().reset_index()
    media_m2_zona = media_m2_zona.sort_values(by='valor_m2', ascending=False)
    return media_m2_zona


if __name__ == "__main__":
    import os

    # Caminho do arquivo
    caminho_arquivo = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "dados_imoveis",
                                   "bairro_final_v3_engineered.xlsx")

    if os.path.exists(caminho_arquivo):
        df = pd.read_excel(caminho_arquivo)

        print("--- Amostra das Zonas Classificadas ---")
        df_com_zonas = classificar_zonas(df)
        print(df_com_zonas[['Bairro', 'Zona']].sample(5))

        print("\n--- Média de Valor por Bairro (Top 5) ---")
        media_bairro = calcular_media_bairro(df_com_zonas)
        print(media_bairro.head())

        print("\n--- Média de Valor por Zona ---")
        media_zona = calcular_media_zona(df_com_zonas)
        print(media_zona)

        print("\n--- Média de Preço do m² por Zona ---")
        media_m2 = calcular_media_m2_zona(df_com_zonas)
        print(media_m2)
    else:
        print(f"Arquivo não encontrado: {caminho_arquivo}")
