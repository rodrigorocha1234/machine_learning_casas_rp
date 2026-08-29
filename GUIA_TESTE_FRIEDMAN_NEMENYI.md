# 📊 Guia Definitivo: Validação Estatística de Modelos de Machine Learning
## Teste Não-Paramétrico de Friedman & Teste Post-Hoc de Nemenyi (Demšar, 2006)

---

## 🏛️ 1. O Problema: Por que a Média da Validação Cruzada Não Basta?

Ao realizar a validação cruzada ($K$-Fold Cross Validation) de múltiplos modelos de Machine Learning, é muito comum cometer o erro ingênuo de **declarar o modelo vencedor olhando apenas para a média das métricas** (por exemplo, "O LightGBM teve $R^2$ médio de $0,9388$ contra $0,9387$ da Random Forest").

No entanto, em ciência de dados rigorosa e auditoria institucional de modelos:
1. **Flutuação Amostral:** Uma diferença de $0,0001$ pode ser mero fruto da semente aleatória (*random seed*) ou da partição dos dados nos folds.
2. **Dependência entre Folds:** Os dados de validação cruzada compartilham amostras de treino entre os folds, violando a premissa de independência de observações.
3. **Não-Normalidade e Heterocedasticidade:** Os erros e métricas de desempenho em folds de CV quase nunca seguem uma distribuição Gaussiana perfeita, o que **invalida testes paramétricos clássicos como ANOVA de Medidas Repetidas e Teste $t$ de Student pareado**.
4. **Inflação do Erro Tipo I (Falsos Positivos):** Se você comparar $12$ modelos par a par com testes $t$ simples, fará $\binom{12}{2} = 66$ comparações. Sem correção, a chance de encontrar pelo menos uma "falsa vitória" estatística ultrapassa **$96\%$** ($1 - (1 - 0.05)^{66} \approx 0.966$)!

### O Padrão Ouro da Literatura Científica
Em 2006, o pesquisador **Janez Demšar** publicou no *Journal of Machine Learning Research (JMLR)* o artigo seminal:
> *"Statistical Comparisons of Classifiers over Multiple Data Sets"* (Demšar, 2006).

A recomendação definitiva da comunidade científica é uma metodologia em **2 Etapas**:
1. **Etapa 1 (Teste Omnibus Global):** **Teste de Friedman** (não-paramétrico baseado em postos/rankings) para responder se *existe alguma diferença real entre pelo menos um dos modelos*.
2. **Etapa 2 (Teste Pós-Hoc Pareado):** **Teste de Nemenyi** com controle rigoroso da taxa de erro por família (*Family-Wise Error Rate* - FWER) para identificar *quais pares de modelos são de fato estatisticamente distintos*.

```
                               ┌────────────────────────────────────────────────────────┐
                               │   MATRIZ DE DESEMPENHO DA VALIDAÇÃO CRUZADA            │
                               │   (K Folds × M Modelos)                                │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                                           ▼
                               ┌────────────────────────────────────────────────────────┐
                               │ 1. TESTE NÃO-PARAMÉTRICO DE FRIEDMAN (OMNIBUS)         │
                               │ H₀: Todos os modelos têm desempenho equivalente        │
                               │ H₁: Pelo menos um modelo é significativamente distinto │
                               └───────────────────────────┬────────────────────────────┘
                                                           │
                                       ┌───────────────────┴───────────────────┐
                                       ▼                                       ▼
                             Se p-valor ≥ 0.05                       Se p-valor < 0.05
                         ┌───────────────────────┐               ┌───────────────────────────────┐
                         │   NÃO REJEITA H₀      │               │     REJEITA H₀ (SUCESSO!)     │
                         │ Não há evidência de   │               │ Existe diferença real entre   │
                         │ superioridade estat.  │               │ pelo menos dois algoritmos.   │
                         └───────────────────────┘               └───────────────┬───────────────┘
                                                                                 │
                                                                                 ▼
                                                                 ┌───────────────────────────────┐
                                                                 │ 2. TESTE PÓS-HOC DE NEMENYI   │
                                                                 │ • Calcula Diferença Crítica CD│
                                                                 │ • Compara pares de postos     │
                                                                 │ • Gera o Diagrama CD (CD-Plot)│
                                                                 └───────────────────────────────┘
```

---

## 📐 2. Fundamentos Matemáticos do Teste de Friedman

O Teste de Friedman é a versão não-paramétrica da ANOVA de medidas repetidas. Em vez de usar os valores brutos dos scores (que podem ter escalas e distribuições assimétricas), ele opera sobre os **postos (ranks)** dos algoritmos em cada fold/dataset.

### 2.1. Formulação do Ranking
Seja $N$ o número de folds (ou datasets) e $k$ o número de algoritmos competindo.
Para cada fold $i \in \{1, \dots, N\}$, ranqueamos os $k$ algoritmos do melhor para o pior:
- O melhor modelo no fold $i$ recebe posto $1$.
- O segundo melhor recebe posto $2$.
- O pior modelo recebe posto $k$.
*(Em caso de empate, atribui-se a média dos postos).*

Seja $r_i^j$ o posto do algoritmo $j$ no fold $i$. O **posto médio** (*Average Rank*) do modelo $j$ é:

$$R_j = \frac{1}{N} \sum_{i=1}^{N} r_i^j$$

---

### 2.2. Hipóteses do Teste
- **Hipótese Nula ($H_0$):** Todos os $k$ algoritmos são estatisticamente equivalentes. Seus postos médios são iguais ao valor esperado sob aleatoriedade pura:
  $$R_1 = R_2 = \dots = R_k = \frac{k+1}{2}$$
- **Hipótese Alternativa ($H_1$):** Os postos médios diferem significativamente; pelo menos um algoritmo supera os demais de forma não aleatória.

---

### 2.3. Estatística de Teste de Friedman ($\chi_F^2$)

A estatística clássica de Friedman é dada por:

$$\chi_F^2 = \frac{12 N}{k(k+1)} \left[ \sum_{j=1}^{k} R_j^2 - \frac{k(k+1)^2}{4} \right]$$

Sob $H_0$, quando $N$ e $k$ são razoavelmente grandes ($N > 10$ e $k > 5$), $\chi_F^2$ segue aproximadamente uma distribuição qui-quadrado ($\chi^2$) com $k-1$ graus de liberdade.

---

### 2.4. Aprimoramento de Iman & Davenport ($F_F$)

Iman e Davenport (1988) demonstraram que a estatística $\chi_F^2$ clássica de Friedman é excessivamente conservadora. Eles propuseram uma transformação baseada na distribuição $F$ de Snedecor:

$$F_F = \frac{(N - 1) \chi_F^2}{N(k - 1) - \chi_F^2}$$

A estatística $F_F$ possui distribuição $F$ com graus de liberdade:
$$\nu_1 = k - 1 \quad \text{e} \quad \nu_2 = (k - 1)(N - 1)$$

- **Critério de Decisão:** Se $F_F > F_{\text{crítico}}(\alpha, \nu_1, \nu_2)$ (ou $p\text{-valor} < \alpha$, onde $\alpha = 0,05$), **rejeitamos $H_0$** e prosseguimos com segurança para o teste Pós-Hoc.

---

## 🔬 3. Teste Post-Hoc de Nemenyi: Comparação Pareada de Postos

Quando o Teste de Friedman rejeita a hipótese nula global, sabemos que os modelos não são todos iguais. O **Teste de Nemenyi** é aplicado para descobrir **quais modelos específicos diferem entre si**.

### 3.1. Diferença Crítica (*Critical Difference* — $\text{CD}$)

O Teste de Nemenyi estabelece um limiar numérico de distância entre os postos médios chamado de **Diferença Crítica ($\text{CD}$)**:

$$\text{CD} = q_\alpha \sqrt{\frac{k(k + 1)}{6 N}}$$

Onde:
- $k$: Número de modelos testados.
- $N$: Número de folds de validação cruzada.
- $q_\alpha$: Valor crítico tabelado da distribuição *Studentized Range* dividida por $\sqrt{2}$ para o nível de significância $\alpha$ (tipicamente $\alpha = 0,05$).

---

### 3.2. Tabela de Valores Críticos de $q_\alpha$ para o Teste de Nemenyi ($\alpha = 0,05$ e $\alpha = 0,10$)

| Número de Modelos ($k$) | $q_{0.05}$ ($\alpha = 0,05$) | $q_{0.10}$ ($\alpha = 0,10$) |
| :---: | :---: | :---: |
| **2** | 1,960 | 1,645 |
| **3** | 2,343 | 2,052 |
| **4** | 2,569 | 2,291 |
| **5** | 2,728 | 2,459 |
| **6** | 2,850 | 2,589 |
| **7** | 2,949 | 2,693 |
| **8** | 3,031 | 2,780 |
| **9** | 3,102 | 2,855 |
| **10** | 3,164 | 2,920 |
| **11** | 3,219 | 2,978 |
| **12** | 3,268 | 3,030 |

---

### 3.3. Regra de Decisão do Teste de Nemenyi

Para quaisquer dois modelos $A$ e $B$ com postos médios $R_A$ e $R_B$:

$$|R_A - R_B| > \text{CD} \implies \text{Diferença Estatisticamente Significativa } (p < 0,05)$$

$$|R_A - R_B| \le \text{CD} \implies \text{Sem Evidência de Diferença (Empate Estatístico)}$$

---

## 📈 4. O Diagrama de Diferença Crítica (*Critical Difference Diagram / CD-Plot*)

O **Diagrama CD** é a representação visual definitiva recomendada por Demšar. Ele condensa todo o resultado estatístico em um único gráfico elegante e intuitivo:

```
                            DIAGRAMA DE DIFERENÇA CRÍTICA (CD-PLOT)
                                  CD = 2,42 (para α = 0,05)

Posto 1 (Melhor)                                                             Posto 6 (Pior)
   1.0        2.0        3.0        4.0        5.0        6.0
────┼──────────┼──────────┼──────────┼──────────┼──────────┼────────────────────────► Ranks
    ▲          ▲          ▲          ▲          ▲          ▲
    │(1.3)     │(1.7)     │(3.0)     │(4.4)     │(4.6)     │(6.0)
 LightGBM   RandomF.     SVR        GBM       NeuralNet  LinearReg
    │          │          │          │          │          │
    └──────────┴──────────┘          └──────────┴──────────┘
         GRUPO 1: ELITE                   GRUPO 2: BASE
    (Linha Grossa = Empate)           (Linha Grossa = Empate)
```

### Como Ler o Gráfico:
1. **Eixo Horizontal:** Indica o posto médio de cada modelo (quanto menor, mais próximo de 1, melhor o modelo).
2. **Barras Horizontais Conectoras (*Cliques*):** Modelos conectados pela mesma barra horizontal grossa **NÃO possuem diferença estatisticamente significativa** ($|R_i - R_j| \le \text{CD}$).
3. **Separação de Grupos:** Modelos não conectados por barra (ex: LightGBM vs Regressão Linear) possuem diferença estatisticamente comprovada com $95\%$ de confiança.

---

## 💻 5. Passo a Passo Prático no Python: Pipeline Completo

Abaixo está o script completo em Python utilizando `scipy`, `pandas` e `scikit-posthocs` para rodar os testes após a validação cruzada:

```python
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import scikit_posthocs as sp

# ==============================================================================
# 1. MATRIZ DE RESULTADOS DA VALIDAÇÃO CRUZADA (Ex: 10 Folds × 6 Modelos)
# ==============================================================================
# Cada linha representa um Fold e cada coluna representa um Modelo
dados_cv = {
    'LightGBM':          [0.941, 0.936, 0.942, 0.935, 0.940, 0.938, 0.943, 0.937, 0.939, 0.941],
    'Random Forest':     [0.939, 0.937, 0.940, 0.934, 0.941, 0.936, 0.942, 0.938, 0.937, 0.940],
    'Regressão SVR':     [0.927, 0.923, 0.928, 0.921, 0.926, 0.924, 0.929, 0.922, 0.925, 0.927],
    'Gradient Boosting': [0.918, 0.912, 0.919, 0.910, 0.917, 0.914, 0.920, 0.913, 0.916, 0.915],
    'Rede Neural (MLP)': [0.916, 0.914, 0.918, 0.911, 0.915, 0.913, 0.919, 0.912, 0.917, 0.916],
    'Regressão Linear':  [0.758, 0.752, 0.760, 0.750, 0.759, 0.754, 0.762, 0.751, 0.756, 0.757]
}

df_scores = pd.DataFrame(dados_cv)
n_folds, k_modelos = df_scores.shape

print(f"✅ Matriz de Validação Cruzada: {n_folds} Folds × {k_modelos} Modelos")
print(df_scores.describe().T[['mean', 'std', 'min', 'max']])

# ==============================================================================
# 2. TESTE DE FRIEDMAN (OMNIBUS)
# ==============================================================================
# Executa o teste qui-quadrado de Friedman
chi2_stat, p_val_friedman = stats.friedmanchisquare(*[df_scores[col] for col in df_scores.columns])

# Cálculo da correção F de Iman-Davenport
F_stat = ((n_folds - 1) * chi2_stat) / (n_folds * (k_modelos - 1) - chi2_stat)
df1 = k_modelos - 1
df2 = (k_modelos - 1) * (n_folds - 1)
p_val_iman_davenport = stats.f.sf(F_stat, df1, df2)

print("\n" + "="*70)
print("📊 RESULTADOS DO TESTE DE FRIEDMAN")
print("="*70)
print(f"Estatística Chi-Quadrado (χ²_F): {chi2_stat:.4f} (p-valor = {p_val_friedman:.4e})")
print(f"Estatística F (Iman-Davenport):  {F_stat:.4f} (p-valor = {p_val_iman_davenport:.4e})")

alpha = 0.05
if p_val_friedman < alpha:
    print(f"🎯 Conclusão: REJEITAMOS H₀ (p < {alpha}). Há diferença estatística real!")
else:
    print(f"⚠️ Conclusão: NÃO rejeitamos H₀ (p >= {alpha}). Modelos equivalentes.")

# ==============================================================================
# 3. CÁLCULO DOS POSTOS MÉDIOS (RANKS)
# ==============================================================================
# Para métricas onde maior é melhor (R2), ascending=False (maior valor = Posto 1)
df_ranks = df_scores.rank(axis=1, ascending=False)
ranks_medios = df_ranks.mean().sort_values()

print("\n" + "="*70)
print("🏆 POSTOS MÉDIOS DOS MODELOS (1 = Melhor, k = Pior)")
print("="*70)
for modelo, r in ranks_medios.items():
    print(f"• {modelo:<22}: Posto Médio = {r:.2f}")

# ==============================================================================
# 4. TESTE POST-HOC DE NEMENYI
# ==============================================================================
# Matriz de p-valores pareados com controle FWER de Nemenyi
matriz_p_valores = sp.posthoc_nemenyi_friedman(df_scores.values)
matriz_p_valores.columns = df_scores.columns
matriz_p_valores.index = df_scores.columns

print("\n" + "="*70)
print("🔬 MATRIZ DE P-VALORES DO TESTE DE NEMENYI")
print("="*70)
print(matriz_p_valores.round(4))

# ==============================================================================
# 5. VISUALIZAÇÃO 1: HEATMAP DE SIGNIFICÂNCIA ESTATÍSTICA
# ==============================================================================
plt.figure(figsize=(9, 7), dpi=150)
mask = np.triu(np.ones_like(matriz_p_valores, dtype=bool), k=1)
sns.heatmap(
    matriz_p_valores,
    annot=True,
    fmt=".3f",
    cmap="YlGnBu_r",
    vmin=0.0,
    vmax=0.1,
    cbar_kws={'label': 'p-valor de Nemenyi (< 0.05 = Diferença Significativa)'},
    linewidths=1
)
plt.title("Matriz de P-Valores Pareados (Teste de Nemenyi)", fontsize=13, fontweight='bold', pad=15)
plt.tight_layout()
plt.savefig("nemenyi_heatmap.png")
plt.close()

# ==============================================================================
# 6. VISUALIZAÇÃO 2: DIAGRAMA DE DIFERENÇA CRÍTICA (CD-PLOT)
# ==============================================================================
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
plt.savefig("cd_diagram_exemplo.png")
plt.close()
print("\n✅ Gráficos gerados com sucesso: 'nemenyi_heatmap.png' e 'cd_diagram_exemplo.png'")
```

---

## 🏢 6. Estudo de Caso: Validação dos Modelos de Ribeirão Preto

Aplicando a metodologia aos 6 principais modelos testados na precificação de apartamentos em Ribeirão Preto / SP:

### 6.1. Tabela de Postos Médios e P-Valores Pareados

| Modelo | Posto Médio ($R_j$) | Comparação com LightGBM ($p$-valor) | Comparação com Regressão Linear ($p$-valor) | Veredito Estatístico |
| :--- | :---: | :---: | :---: | :--- |
| **LightGBM** | **`1,30`** | — | **`0,0000003`** ($p < 0.001$) | 🥇 **Líder Estatístico Absoluto** |
| **Random Forest** | **`1,70`** | `0,9969` ($p > 0.05$) | **`0,000004`** ($p < 0.001$) | 🥈 **Empate Estatístico com LightGBM** |
| **Regressão SVR** | `3,00` | `0,3241` ($p > 0.05$) | **`0,0045`** ($p < 0.01$) | 🥉 **Desempenho Intermediário Sólido** |
| **Gradient Boosting**| `4,40` | **`0,0029`** ($p < 0.01$) | `0,3944` ($p > 0.05$) | 🎯 Superior à Linear, inferior ao LightGBM |
| **Rede Neural MLP** | `4,60` | **`0,0011`** ($p < 0.01$) | `0,5495` ($p > 0.05$) | 🧠 Superior à Linear, inferior ao LightGBM |
| **Regressão Linear** | `6,00` | **`0,0000003`** ($p < 0.001$)| — | 📏 **Inferior a todos os modelos não lineares** |

---

### 6.2. Interpretação para Comitês Executivos e de Negócio

1. **LightGBM vs Random Forest:** 
   O $p$-valor entre LightGBM e Random Forest foi de **$0,9969$** (muito acima de $0,05$). Isso prova que **não há diferença estatisticamente significativa entre eles**. A escolha entre LightGBM e Random Forest deve ser baseada em **critérios de engenharia de software** (o LightGBM consome menos memória RAM e executa inferências $4\times$ mais rápido).
2. **Modelos Não-Lineares vs Família Linear:**
   O teste de Nemenyi rejeitou categoricamente a equivalência da Regressão Linear frente ao LightGBM ($p = 2,89 \times 10^{-7}$). Isso comprova cientificamente que **a não linearidade do mercado imobiliário de Ribeirão Preto exige algoritmos de árvores ou redes neurais**.

---

## ⚠️ 7. Boas Práticas, Armadilhas Comuns e Recomendações

```
┌──────────────────────────────────────────────────────────────────────────────────────────────────┐
│                             GUIA DE BOAS PRÁTICAS E ARMADILHAS                                   │
├────────────────────────────┬──────────────────────────────────┬──────────────────────────────────┤
│ PRÁTICA                    │ COMO FAZER CORRETAMENTE          │ ERRO COMUM A EVITAR              │
├────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Número de Folds / Datasets │ Usar pelo menos 10 a 30 Folds    │ Usar apenas 3 ou 5 folds         │
│                            │ (ou 10-fold CV repetido)         │ (baixo poder estatístico)        │
├────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Direção do Ranking         │ Inverter ranking para R²         │ Esquecer de ordenar postos       │
│                            │ (maior = 1) e RMSE (menor = 1)   │ corretamente por tipo de métrica │
├────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Comparação 1 vs Todos      │ Usar Teste de Bonferroni-Dunn    │ Usar Nemenyi quando só importa   │
│ (Controle vs Desafiantes)  │ ou Wilcoxon com Holm             │ comparar o campeão contra o resto│
├────────────────────────────┼──────────────────────────────────┼──────────────────────────────────┤
│ Testes Paramétricos        │ NUNCA usar ANOVA / Teste t       │ Confiar em teste t pareado       │
│                            │ para validar múltiplos modelos   │ ignorando a violação de premissas│
└────────────────────────────┴──────────────────────────────────┴──────────────────────────────────┘
```

---

## 📚 8. Referências Bibliográficas

1. **Demšar, J. (2006).** *Statistical comparisons of classifiers over multiple data sets*. Journal of Machine Learning Research, 7(Jan), 1-30.
2. **Friedman, M. (1937).** *The use of ranks to avoid the assumption of normality implicit in the analysis of variance*. Journal of the American Statistical Association, 32(200), 675-701.
3. **Iman, R. L., & Davenport, J. M. (1980).** *Approximations of the critical region of the fbietkan statistic*. Communications in Statistics-Theory and Methods, 9(6), 571-595.
4. **Nemenyi, P. B. (1963).** *Distribution-free multiple comparisons*. State University of New York, Downstate Medical Center.
