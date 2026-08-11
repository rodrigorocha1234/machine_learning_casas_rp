import pandas as pd

df = pd.read_excel('dados_imoveis/bairro_final_v2.xlsx')
print(df.info())
print("\nFirst 5 rows:")
print(df.head())
