#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Script de Avaliação Estatística Não-Paramétrica de Modelos de Machine Learning
Metodologia: Teste de Friedman (Omnibus) + Teste Post-Hoc de Nemenyi (Demšar, 2006)
"""

import sys
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

try:
    import scikit_posthocs as sp
except ImportError:
    print("❌ scikit-posthocs não instalado. Instale com: pip install scikit-posthocs")
    sys.exit(1)


def executar_analise_estatistica(df_scores: pd.DataFrame, metrica_nome: str = "R2 Score", maior_e_melhor: bool = True, output_dir: str = "."):
    """
    Executa a bateria de testes de Friedman e Nemenyi para uma matriz de validação cruzada.
    
    Args:
        df_scores: DataFrame onde cada linha é um Fold e cada coluna é um Modelo.
        metrica_nome: Nome da métrica avaliada (ex: R2 Score, RMSE, MAE).
        maior_e_melhor: True se maior valor significa melhor modelo (ex: R2), False se menor é melhor (ex: RMSE, MAE).
        output_dir: Diretório para salvar os gráficos gerados.
    """
    n_folds, k_modelos = df_scores.shape
    os.makedirs(output_dir, exist_ok=True)
    
    print("\n" + "═"*75)
    print(f"📊 AVALIAÇÃO ESTATÍSTICA: {k_modelos} MODELOS EM {n_folds} FOLDS ({metrica_nome})")
    print("═"*75)
    
    # 1. Teste de Friedman
    chi2_stat, p_val_friedman = stats.friedmanchisquare(*[df_scores[col] for col in df_scores.columns])
    
    # Correção F de Iman-Davenport
    F_stat = ((n_folds - 1) * chi2_stat) / (n_folds * (k_modelos - 1) - chi2_stat)
    df1 = k_modelos - 1
    df2 = (k_modelos - 1) * (n_folds - 1)
    p_val_iman_davenport = stats.f.sf(F_stat, df1, df2)
    
    print(f"\n1. TESTE DE FRIEDMAN (OMNIBUS):")
    print(f"   • Estatística Chi-Quadrado (χ²_F): {chi2_stat:.4f} (p-valor = {p_val_friedman:.4e})")
    print(f"   • Estatística F (Iman-Davenport):  {F_stat:.4f} (p-valor = {p_val_iman_davenport:.4e})")
    
    alpha = 0.05
    if p_val_friedman < alpha:
        print(f"   🎯 Conclusão: REJEITAMOS H₀ (p < {alpha}). Diferença estatística comprovada!")
    else:
        print(f"   ⚠️ Conclusão: NÃO rejeitamos H₀ (p >= {alpha}). Desempenhos equivalentes.")
        return
    
    # 2. Ranking dos Modelos
    # Se maior é melhor (R2), maior valor ganha rank 1 (ascending=False)
    # Se menor é melhor (RMSE), menor valor ganha rank 1 (ascending=True)
    df_ranks = df_scores.rank(axis=1, ascending=not maior_e_melhor)
    ranks_medios = df_ranks.mean().sort_values()
    
    print(f"\n2. POSTOS MÉDIOS (RANKS) - (1 = Melhor Desempenho):")
    for pos, (modelo, r) in enumerate(ranks_medios.items(), 1):
        print(f"   {pos:2d}º. {modelo:<25}: Posto Médio = {r:.2f}")
    
    # 3. Teste Post-Hoc de Nemenyi
    matriz_p_valores = sp.posthoc_nemenyi_friedman(df_scores.values)
    matriz_p_valores.columns = df_scores.columns
    matriz_p_valores.index = df_scores.columns
    
    print(f"\n3. MATRIZ DE P-VALORES PAREADOS (TESTE DE NEMENYI):")
    print(matriz_p_valores.round(4).to_string())
    
    # 4. Geração do Heatmap
    plt.figure(figsize=(9, 7), dpi=150)
    sns.heatmap(
        matriz_p_valores,
        annot=True,
        fmt=".3f",
        cmap="YlGnBu_r",
        vmin=0.0,
        vmax=0.1,
        cbar_kws={'label': f'p-valor de Nemenyi (< {alpha} = Diferença Estatística)'},
        linewidths=1
    )
    plt.title(f"Matriz de P-Valores Pareados - Nemenyi ({metrica_nome})", fontsize=12, fontweight='bold', pad=15)
    plt.tight_layout()
    heatmap_path = os.path.join(output_dir, "nemenyi_pvalues_heatmap.png")
    plt.savefig(heatmap_path)
    plt.close()
    
    # 5. Geração do Diagrama de Diferença Crítica (CD-Plot)
    fig, ax = plt.subplots(figsize=(10, 3.5), dpi=150)
    sp.critical_difference_diagram(
        ranks=ranks_medios,
        sig_matrix=matriz_p_valores,
        ax=ax,
        label_fmt_left='{label} ({rank:.2f})',
        label_fmt_right='({rank:.2f}) {label}'
    )
    plt.title(f"Diagrama de Diferença Crítica (CD-Plot) - Validação Cruzada {n_folds}-Fold (α={alpha})", 
              fontsize=11, fontweight='bold', pad=20)
    plt.tight_layout()
    cd_path = os.path.join(output_dir, "critical_difference_diagram.png")
    plt.savefig(cd_path)
    plt.close()
    
    print(f"\n📁 Gráficos salvos com sucesso:")
    print(f"   • Heatmap: {heatmap_path}")
    print(f"   • CD-Plot: {cd_path}")
    print("═"*75 + "\n")


if __name__ == "__main__":
    # Exemplo com 10 Folds dos 6 principais modelos avaliados em Ribeirão Preto
    dados_exemplo = {
        'LightGBM':          [0.941, 0.936, 0.942, 0.935, 0.940, 0.938, 0.943, 0.937, 0.939, 0.941],
        'Random Forest':     [0.939, 0.937, 0.940, 0.934, 0.941, 0.936, 0.942, 0.938, 0.937, 0.940],
        'Regressão SVR':     [0.927, 0.923, 0.928, 0.921, 0.926, 0.924, 0.929, 0.922, 0.925, 0.927],
        'Gradient Boosting': [0.918, 0.912, 0.919, 0.910, 0.917, 0.914, 0.920, 0.913, 0.916, 0.915],
        'Rede Neural (MLP)': [0.916, 0.914, 0.918, 0.911, 0.915, 0.913, 0.919, 0.912, 0.917, 0.916],
        'Regressão Linear':  [0.758, 0.752, 0.760, 0.750, 0.759, 0.754, 0.762, 0.751, 0.756, 0.757]
    }
    df = pd.DataFrame(dados_exemplo)
    executar_analise_estatistica(df, metrica_nome="R² Score", maior_e_melhor=True, output_dir="eda")
