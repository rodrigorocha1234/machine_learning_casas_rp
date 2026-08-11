import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import json

# Set aesthetic parameters for plots
sns.set_theme(style="whitegrid")
plt.rcParams["figure.figsize"] = (10, 6)

def run_eda():
    data_path = '../dados_imoveis/bairro_final_v2.xlsx'
    
    print(f"Loading data from {data_path}...")
    df = pd.read_excel(data_path)
    
    # 1. Basic Information
    print("\n--- Basic Information ---")
    print(f"Dataset shape: {df.shape}")
    print("\nMissing values:")
    missing = df.isnull().sum()
    print(missing[missing > 0] if missing.sum() > 0 else "No missing values.")
    
    print("\nData types:")
    print(df.dtypes)
    
    # 2. Descriptive Statistics
    print("\n--- Descriptive Statistics (Numerical) ---")
    desc_stats = df.describe().round(2)
    print(desc_stats)
    
    print("\n--- Descriptive Statistics (Categorical) ---")
    cat_cols = df.select_dtypes(include=['object']).columns
    print(df[cat_cols].describe())
    
    # Save descriptive stats to JSON for the report
    stats_dict = {
        "shape": df.shape,
        "missing": missing.to_dict(),
        "numerical_desc": desc_stats.to_dict()
    }
    with open('eda_stats.json', 'w') as f:
        json.dump(stats_dict, f, default=str, indent=4)
        
    # 3. Univariate Analysis
    print("\n--- Generating Plots ---")
    
    # 3.1 Target Variable Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Valor de Venda (R$)'], kde=True, bins=50)
    plt.title('Distribuição do Valor de Venda (R$)')
    plt.xlabel('Valor (R$)')
    plt.ylabel('Frequência')
    plt.savefig('dist_valor_venda.png')
    plt.close()
    
    # 3.2 Target Variable Boxplot (Outliers)
    plt.figure(figsize=(10, 5))
    sns.boxplot(x=df['Valor de Venda (R$)'])
    plt.title('Boxplot do Valor de Venda (R$)')
    plt.xlabel('Valor (R$)')
    plt.savefig('box_valor_venda.png')
    plt.close()
    
    # 3.3 Metragem Distribution
    plt.figure(figsize=(10, 5))
    sns.histplot(df['Metragem (m²)'], kde=True, bins=50, color='orange')
    plt.title('Distribuição da Metragem (m²)')
    plt.xlabel('Metragem (m²)')
    plt.ylabel('Frequência')
    plt.savefig('dist_metragem.png')
    plt.close()
    
    # 4. Bivariate Analysis
    # 4.1 Valor vs Metragem Scatterplot
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x='Metragem (m²)', y='Valor de Venda (R$)', data=df, alpha=0.5)
    plt.title('Valor de Venda vs Metragem')
    plt.xlabel('Metragem (m²)')
    plt.ylabel('Valor de Venda (R$)')
    plt.savefig('scatter_valor_metragem.png')
    plt.close()
    
    # 4.2 Boxplot Valor by Zona
    plt.figure(figsize=(12, 6))
    sns.boxplot(x='Zona', y='Valor de Venda (R$)', data=df, order=df['Zona'].value_counts().index)
    plt.title('Valor de Venda por Zona')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig('box_valor_zona.png')
    plt.close()
    
    # 5. Correlation Matrix
    numerical_df = df.select_dtypes(include=['int64', 'float64']).drop(columns=['Código'], errors='ignore')
    plt.figure(figsize=(10, 8))
    corr = numerical_df.corr(method='spearman')
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f", vmin=-1, vmax=1)
    plt.title('Matriz de Correlação (Spearman)')
    plt.tight_layout()
    plt.savefig('heatmap_correlacao.png')
    plt.close()
    
    print("EDA completed. Plots and stats saved in eda/ directory.")

if __name__ == "__main__":
    run_eda()
