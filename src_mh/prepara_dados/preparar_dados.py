import os
from typing import Final, Literal

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler, MinMaxScaler

from src_mh.config.config import Config
from src_mh.prepara_dados.iprepara_dados import IPrepararDados

pd.set_option("display.max_columns", None)  # Exibe todas as colunas
pd.set_option("display.max_rows", 50)  # Máximo de linhas exibidas
pd.set_option("display.width", 200)  # Largura do DataFrame
pd.set_option("display.max_colwidth", 50)  # Largura máxima das colunas

pd.set_option("display.expand_frame_repr", False)  # Evita quebrar o DataFrame


class PrepararDadosDataFame(IPrepararDados[pd.DataFrame]):
    __MAPA_ZONAS: Final[dict[str, str]] = {
        'Adao do Carmo Leonel': 'Zona Norte', 'Adelino Simioni': 'Zona Norte',
        'Alamedas do Botanico': 'Zona Sul', 'Alto da Boa Vista': 'Zona Oeste',
        'Alto do Ipiranga': 'Zona Norte',
        'Bonfim Paulista': 'Zona Sul', 'Bosque das Juritis': 'Zona Sul',
        'Campos Eliseos': 'Zona Norte',
        'Castelo Branco': 'Zona Leste', 'Centro': 'Centro',
        'Chacaras Bonacorsi': 'Zona Leste',
        'Chacaras Pedro Correa de Carvalho': 'Zona Leste',
        'City Ribeirao': 'Zona Leste',
        'Condominio Itamaraty': 'Zona Sul', 'Condominio Mirante Sul': 'Zona Sul',
        'Condominio Uirapuru': 'Zona Sul',
        'Condominio Vila Florenca': 'Zona Sul',
        'Conjunto Habitacional Jardim das Palmeiras': 'Zona Leste',
        'Conjunto Habitacional Silvio Passalacqua': 'Zona Norte',
        'Dom Bernardo Jose Mielle': 'Zona Norte',
        'Geraldo Correia de Carvalho': 'Zona Norte', 'Higienopolis': 'Centro',
        'Iguatemi': 'Zona Leste',
        'Independencia': 'Zona Leste', 'Ipiranga': 'Zona Norte',
        'Jamil Seme Cury': 'Zona Norte',
        'Jardim America': 'Zona Oeste', 'Jardim Angelo Jurca': 'Zona Norte',
        'Jardim Anhanguera': 'Zona Leste',
        'Jardim Antartica': 'Zona Oeste', 'Jardim Bela Vista': 'Zona Oeste',
        'Jardim Bom Pastor': 'Zona Leste',
        'Jardim Botanico': 'Zona Sul', 'Jardim California': 'Zona Sul',
        'Jardim Canada': 'Zona Sul',
        'Jardim Castelo Branco': 'Zona Leste', 'Jardim Centenario': 'Zona Oeste',
        'Jardim Cybelli': 'Zona Norte',
        'Jardim Diva Tarla de Carvalho': 'Zona Norte',
        'Jardim Doutor Paulo Gomes Romeo': 'Zona Oeste',
        'Jardim Eugenio Mendes Lopes': 'Zona Norte',
        'Jardim Florestan Fernandes': 'Zona Norte',
        'Jardim Florida': 'Zona Sul', 'Jardim Formoso': 'Zona Oeste',
        'Jardim Guapore': 'Zona Sul',
        'Jardim Heitor Rigon': 'Zona Norte', 'Jardim Helena': 'Zona Leste',
        'Jardim Herculano Fernandes': 'Zona Norte',
        'Jardim Ilhas do Sul': 'Zona Sul', 'Jardim Interlagos': 'Zona Leste',
        'Jardim Iraja': 'Zona Sul',
        'Jardim Itapora': 'Zona Norte', 'Jardim Itau': 'Zona Oeste',
        'Jardim Jandaia': 'Zona Norte',
        'Jardim Javari': 'Zona Norte', 'Jardim Joao Rossi': 'Zona Sul',
        'Jardim Jose Figueira': 'Zona Sul',
        'Jardim Jose Sampaio Junior': 'Zona Norte',
        'Jardim Jose Wilson Toni': 'Zona Norte',
        'Jardim Macedo': 'Zona Leste', 'Jardim Manoel Penna': 'Zona Leste',
        'Jardim Marchesi': 'Zona Oeste',
        'Jardim Maria Goretti': 'Zona Oeste', 'Jardim Mosteiro': 'Zona Norte',
        'Jardim Nova Alianca Sul': 'Zona Sul',
        'Jardim Novo Mundo': 'Zona Leste', 'Jardim Olhos DAgua': 'Zona Sul',
        'Jardim Ouro Branco': 'Zona Leste',
        'Jardim Paiva': 'Zona Oeste', 'Jardim Palma Travassos': 'Zona Leste',
        'Jardim Paulista': 'Zona Leste',
        'Jardim Paulistano': 'Zona Leste', 'Jardim Pedra Branca': 'Zona Leste',
        'Jardim Presidente Dutra': 'Zona Norte',
        'Jardim Presidente Dutra II': 'Zona Norte', 'Jardim Recreio': 'Zona Oeste',
        'Jardim Regatas': 'Zona Norte',
        'Jardim Saint Gerard': 'Zona Sul', 'Jardim San Marco': 'Zona Sul',
        'Jardim Santa Cecilia': 'Zona Sul',
        'Jardim Sao Fernando': 'Zona Leste', 'Jardim Sao Jose': 'Zona Leste',
        'Jardim Sao Luiz': 'Zona Sul',
        'Jardim Sumare': 'Zona Sul', 'Jardim Vilico Cantarelli': 'Zona Oeste',
        'Jardim Villarica': 'Zona Sul',
        'Jardim Zaneti': 'Zona Sul', 'Jardim Zara': 'Zona Leste',
        'Jardim das Oliveiras': 'Zona Leste',
        'Jardim das Palmeiras': 'Zona Leste', 'Jardim do Trevo': 'Zona Leste',
        'Lagoinha': 'Zona Leste',
        'Loteamento Santa Marta': 'Zona Sul', 'Nova Alianca': 'Zona Sul',
        'Nova Ribeirania': 'Zona Leste',
        'Panamby II': 'Zona Sul', 'Parque Anhanguera': 'Zona Leste',
        'Parque Industrial Lagoinha': 'Zona Leste',
        'Parque Residencial Candido Portinari': 'Zona Leste',
        'Parque Residencial Lagoinha': 'Zona Leste',
        'Parque Ribeirao Preto': 'Zona Oeste', 'Parque Sao Sebastiao': 'Zona Leste',
        'Parque das Oliveiras': 'Zona Norte', 'Parque das Oliveiras II': 'Zona Norte',
        'Parque dos Bandeirantes': 'Zona Leste', 'Parque dos Lagos': 'Zona Leste',
        'Parque dos Pinus': 'Zona Norte',
        'Planalto Verde': 'Zona Oeste', 'Plazas de Espana': 'Zona Sul',
        'Presidente Medici': 'Zona Leste',
        'Quinta da Primavera': 'Zona Sul', 'Quintas de Sao Jose': 'Zona Sul',
        'Quintino Facci II': 'Zona Norte',
        'Real Sul': 'Zona Sul', 'Recanto das Palmeiras': 'Zona Leste',
        'Recreio Anhanguera': 'Zona Leste',
        'Recreio das Acacias': 'Zona Leste', 'Republica': 'Zona Oeste',
        'Reserva Macauba': 'Zona Norte',
        'Reserva Real': 'Zona Sul', 'Reserva Sao Jose': 'Zona Leste',
        'Reserva Sul Condominio': 'Zona Sul',
        'Residencial Alto do Ipe': 'Zona Sul', 'Residencial Florida': 'Zona Sul',
        'Residencial Greenville': 'Zona Leste', 'Residencial Jequitiba': 'Zona Sul',
        'Residencial Monterrey': 'Zona Leste', 'Residencial Morro do Ipe': 'Zona Sul',
        'Residencial Parque dos Servidores': 'Zona Leste',
        'Residencial Taiwan': 'Zona Sul',
        'Residencial das Americas': 'Zona Leste',
        'Residencial e Comercial Palmares': 'Zona Leste',
        'Ribeirania': 'Zona Leste', 'Ribeirao Verde': 'Zona Norte',
        'Santa Cruz do Jose Jacques': 'Zona Sul',
        'Setor Central': 'Centro', 'Sumarezinho': 'Zona Oeste',
        'Valentina Figueiredo': 'Zona Norte',
        'Vila Abranches': 'Zona Leste', 'Vila Albertina': 'Zona Norte',
        'Vila Amelia': 'Zona Oeste',
        'Vila Ana Maria': 'Zona Sul', 'Vila Elisa': 'Zona Norte',
        'Vila Guiomar': 'Zona Oeste',
        'Vila Maria Luiza': 'Zona Sul', 'Vila Mariana': 'Zona Norte',
        'Vila Monte Alegre': 'Zona Oeste',
        'Vila Recreio': 'Zona Norte', 'Vila Seixas': 'Centro',
        'Vila Tamandare': 'Zona Norte',
        'Vila Tiberio': 'Zona Oeste', 'Vila Virginia': 'Zona Oeste',
        'Vila do Golf': 'Zona Sul'
    }


    def __classificar_zonas(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Classifica os bairros em zonas da cidade.
        Cria uma nova coluna 'Zona' baseada na coluna 'Bairro'.
        """
        df_copy = df.copy()

        if 'Bairro' not in df_copy.columns:
            raise ValueError("O DataFrame deve conter a coluna 'Bairro'.")

        df_copy['Zona'] = df_copy['Bairro'].map(self.__MAPA_ZONAS).fillna('Centro/Outros')
        return df_copy

    def __calcular_media_m2_zona(self, df: pd.DataFrame) -> pd.DataFrame:
        """Calcula o valor do m² e a média do m² por Zona,

        mantendo TODAS as colunas originais do DataFrame.
        """
        if "Valor_da_Venda" not in df.columns or "Metragem" not in df.columns:
            raise ValueError(
                "O DataFrame deve conter as colunas 'Valor_da_Venda' e"
                " 'Metragem'."
            )

        # Copia o DataFrame recebido para preservar as colunas originais
        df_m2 = df.copy()

        # 1. Cria a coluna de valor do m2 do imóvel individual
        df_m2["Valor_m2"] = df_m2["Valor_da_Venda"] / df_m2["Metragem"]




        # 2. Mapeia a média do m² por Zona de volta para o DataFrame mantendo todas as linhas/colunas
        df_m2["Media_m2_Zona"] = df_m2.groupby("Zona")["Valor_m2"].transform(
            "mean"
        )

        return df_m2

    def realizar_engenharia_atributos(self, df: pd.DataFrame) -> pd.DataFrame:
        # Passos manuais que necessitam acesso à base inteira e à variável alvo (Target Encoding)
        df_copy = self.__classificar_zonas(df)
        df_copy = self.__calcular_media_m2_zona(df_copy)
        
        # Remove colunas que podem causar Data Leakage ou dimensionalidade desnecessária
        df_copy.drop(columns=['Bairro', 'Valor_m2', 'Código'], inplace=True, errors='ignore')
        
        return df_copy
        
    def construir_pipeline(self, categorical_cols: list, numerical_cols: list, tipo_escalonamento: Literal["standard", "minmax", None]) -> Pipeline:
        transformers = []
        
        # 1. Tratamento de Categóricas
        if categorical_cols:
            transformers.append(('cat', OneHotEncoder(sparse_output=False, drop='first'), categorical_cols))
            
        # 2. Tratamento de Numéricas (Escalonamento)
        if tipo_escalonamento:
            if tipo_escalonamento.lower() == 'standard':
                scaler = StandardScaler()
            elif tipo_escalonamento.lower() == 'minmax':
                scaler = MinMaxScaler()
            else:
                raise ValueError("tipo_escalonamento deve ser 'standard' ou 'minmax'")
            transformers.append(('num', scaler, numerical_cols))
        else:
            # Se não houver escalonamento, apenas passa os números adiante
            transformers.append(('num', 'passthrough', numerical_cols))
            
        # Monta o preprocessor mestre
        preprocessor = ColumnTransformer(transformers=transformers, remainder='drop')
        preprocessor.set_output(transform="pandas")
        
        return Pipeline(steps=[('preprocessor', preprocessor)])

    def separar_treino_teste(self, df_final: pd.DataFrame, tipo_escalonamento: Literal["standard", "minmax", None]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

        x = df_final.drop(columns='Valor_da_Venda')
        y = df_final['Valor_da_Venda']

        x_train, x_test, y_train, y_test = train_test_split(
            x,
            y,
            test_size=Config.DADOS_TESTE,
            random_state=Config.RANDOM_STATE
        )

        categorical_cols = x_train.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = x_train.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        # 1. Instancia o pipeline oficial
        pipeline = self.construir_pipeline(categorical_cols, numerical_cols, tipo_escalonamento)
        
        # 2. Fit and Transform exclusivo no x_train (evita leakage)
        x_train = pipeline.fit_transform(x_train)
        
        # 3. Apenas Transform no x_test
        x_test = pipeline.transform(x_test)
        
        # Limpar o prefixo 'cat__' ou 'num__' que o ColumnTransformer adiciona aos DataFrames
        x_train.columns = [col.split('__')[-1] for col in x_train.columns]
        x_test.columns = [col.split('__')[-1] for col in x_test.columns]

        return x_train, x_test, y_train, y_test
