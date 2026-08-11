import pandas as pd
import os

def run_feature_engineering():
    input_path = 'dados_imoveis/bairro_final_v2.xlsx'
    output_path = 'dados_imoveis/bairro_final_v3_engineered.xlsx'
    
    print(f"Lendo dados de {input_path}...")
    df = pd.read_excel(input_path)
    
    # 1. Criar a variável valor_m2 individual para cada imóvel
    print("Criando feature: valor_m2 (Valor / Metragem)...")
    df['valor_m2'] = df['Valor de Venda (R$)'] / df['Metragem (m²)']
    
    # 2. Calcular o valor médio do m2 por Bairro (Target Encoding / Agrupamento Histórico)
    print("Calculando o valor médio do m² por bairro...")
    media_bairro = df.groupby('Bairro')['valor_m2'].mean().reset_index()
    media_bairro.rename(columns={'valor_m2': 'media_valor_m2_bairro'}, inplace=True)
    
    print("Calculando o valor médio do m² por zona...")
    media_zona = df.groupby('Zona')['valor_m2'].mean().reset_index()
    media_zona.rename(columns={'valor_m2': 'media_valor_m2_zona'}, inplace=True)
    
    # 3. Fazer o merge da média histórica de volta no dataframe original
    df = df.merge(media_bairro, on='Bairro', how='left')
    df = df.merge(media_zona, on='Zona', how='left')
    
    # Mostrar um preview das novas colunas
    print("\nPreview das novas colunas para os 5 primeiros imóveis:")
    cols_preview = ['Bairro', 'Zona', 'Valor de Venda (R$)', 'Metragem (m²)', 'valor_m2', 'media_valor_m2_bairro', 'media_valor_m2_zona']
    print(df[cols_preview].head())
    
    # 4. Salvar o novo dataset pronto para a modelagem
    print(f"\nSalvando novo dataset com features em {output_path}...")
    df.to_excel(output_path, index=False)
    print("Concluído!")

if __name__ == "__main__":
    run_feature_engineering()
