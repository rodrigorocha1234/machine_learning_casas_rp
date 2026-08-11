import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Set aesthetic parameters for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def run_eda_engineered():
    data_path = '../dados_imoveis/bairro_final_v3_engineered.xlsx'
    
    print(f"Loading engineered data from {data_path}...")
    df = pd.read_excel(data_path)
    
    print("\n--- Generating Plots for New Features ---")
    
    # 1. Distribution of media_valor_m2_bairro
    plt.figure(figsize=(10, 5))
    sns.histplot(df['media_valor_m2_bairro'], kde=True, bins=50, color='purple')
    plt.title('Distribuição da Média de Valor do m² por Bairro')
    plt.xlabel('Média Valor/m² (R$)')
    plt.ylabel('Frequência')
    plt.savefig('dist_media_bairro.png')
    plt.close()
    
    # 2. Scatter: media_valor_m2_bairro vs Valor de Venda
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='media_valor_m2_bairro', y='Valor de Venda (R$)', data=df, alpha=0.5)
    plt.title('Média do Valor/m² no Bairro vs Valor de Venda')
    plt.xlabel('Média Valor/m² no Bairro (R$)')
    plt.ylabel('Valor de Venda (R$)')
    plt.savefig('scatter_media_bairro_venda.png')
    plt.close()
    
    # 3. Distribution of media_valor_m2_zona
    plt.figure(figsize=(10, 5))
    sns.boxplot(x='Zona', y='media_valor_m2_zona', data=df, order=df['Zona'].value_counts().index)
    plt.title('Média de Valor do m² por Zona')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('box_media_zona.png')
    plt.close()
    
    # 4. Correlation Matrix including new features
    # Excluindo valor_m2 porque ele vaza os dados reais
    numerical_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Código', 'valor_m2'], errors='ignore')
    plt.figure(figsize=(12, 10))
    corr = numerical_df.corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Matriz de Correlação (Spearman) com Novas Features')
    plt.tight_layout()
    plt.savefig('heatmap_correlacao_engineered.png')
    plt.close()
    
    print("EDA completed for engineered features. Plots saved in eda/ directory.")

if __name__ == "__main__":
    run_eda_engineered()
