import pandas as pd
import matplotlib.pyplot as plt
import seaborn as plt_sns
import os
import seaborn as sns
import numpy as np

# Configurar o estilo dos gráficos
sns.set_theme(style="whitegrid")

# Criar diretório para salvar gráficos
output_dir = 'eda/plots'
os.makedirs(output_dir, exist_ok=True)

# Carregar os carregador_dados
df = pd.read_excel('dados_imoveis/bairro_final_v3_engineered.xlsx')

print("1. Estatísticas Descritivas:")
print(df.describe().to_markdown())

# Remover colunas não úteis para modelagem ou que causam vazamento
# Código e Apartamento são identificadores/textos
# valor_m2 causa vazamento (pois é Valor_da_Venda / Metragem)
cols_to_drop = ['Código', 'Apartamento', 'valor_m2']
df_model = df.drop(columns=cols_to_drop)

# Plot 1: Distribuição do Preço (Valor_da_Venda)
plt.figure(figsize=(10, 6))
sns.histplot(df_model['Valor_da_Venda'], kde=True, bins=50, color='blue')
plt.title('Distribuição do Valor de Venda', fontsize=16)
plt.xlabel('Valor de Venda (R$)', fontsize=14)
plt.ylabel('Frequência', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'dist_valor_venda.png'))
plt.close()

# Plot 2: Matriz de Correlação
plt.figure(figsize=(10, 8))
numeric_df = df_model.select_dtypes(include=[np.number])
corr = numeric_df.corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f', square=True, linewidths=.5)
plt.title('Matriz de Correlação (Variáveis Numéricas)', fontsize=16)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'matriz_correlacao.png'))
plt.close()

# Plot 3: Valor_da_Venda vs Metragem
plt.figure(figsize=(10, 6))
sns.scatterplot(x='Metragem', y='Valor_da_Venda', data=df_model, alpha=0.6, color='darkgreen')
plt.title('Valor de Venda vs Metragem', fontsize=16)
plt.xlabel('Metragem (m²)', fontsize=14)
plt.ylabel('Valor de Venda (R$)', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'preco_vs_metragem.png'))
plt.close()

# Plot 4: Preço por Zona
plt.figure(figsize=(10, 6))
order = df_model.groupby('Zona')['Valor_da_Venda'].median().sort_values(ascending=False).index
sns.boxplot(x='Zona', y='Valor_da_Venda', data=df_model, order=order, palette='Set2')
plt.title('Distribuição do Valor de Venda por Zona', fontsize=16)
plt.xlabel('Zona', fontsize=14)
plt.ylabel('Valor de Venda (R$)', fontsize=14)
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig(os.path.join(output_dir, 'preco_por_zona.png'))
plt.close()

# Plot 5: Relação entre Quartos, Banheiros, Vagas e Preço (Pairplot - Opcional)
# A matriz de correlação já cobre bem.
# Vamos ver a quantidade de cada categoria

print("\n2. Valores por Zona:")
print(df_model['Zona'].value_counts())

print("\nEDA Concluída. Gráficos salvos em eda/plots/")
