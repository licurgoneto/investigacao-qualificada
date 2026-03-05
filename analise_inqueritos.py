# ==============================================================================
# DISSERTAÇÃO DE MESTRADO - ANÁLISE DE DADOS DE INQUÉRITOS POLICIAIS (PCRN)
# Autor: Licurgo Nunes Neto
# Descrição: Este script consolida as análises estatísticas e modelos de 
# Machine Learning aplicados a uma amostra de N=172 inquéritos de homicídio.
# ==============================================================================

# 1. IMPORTAÇÃO DAS BIBLIOTECAS NECESSÁRIAS
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.naive_bayes import BernoulliNB
from sklearn.linear_model import LinearRegression, LogisticRegression
import io
import warnings

warnings.filterwarnings('ignore') # Oculta avisos estéticos do matplotlib/seaborn

# Tenta importar o módulo de upload do Google Colab
try:
    from google.colab import files
    IN_COLAB = True
except ImportError:
    IN_COLAB = False

# ==============================================================================
# 2. CARREGAMENTO DA BASE DE DADOS (CSV OU EXCEL)
# ==============================================================================
print("⏳ Iniciando o ambiente de análise...")

if IN_COLAB:
    print("📁 Por favor, faça o upload do arquivo de metadados (CSV ou Excel):")
    uploaded = files.upload()
    file_name = list(uploaded.keys())[0]
    
    # Detecção automática do formato do arquivo
    if file_name.endswith('.csv'):
        df = pd.read_csv(io.BytesIO(uploaded[file_name]))
        if df.columns[0].startswith('Unnamed'):
            df = pd.read_csv(io.BytesIO(uploaded[file_name]), header=1)
    elif file_name.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(io.BytesIO(uploaded[file_name]))
        if df.columns[0].startswith('Unnamed'):
            df = pd.read_excel(io.BytesIO(uploaded[file_name]), header=1)
else:
    # Caso rode localmente no Jupyter Notebook/VS Code
    file_name = input("Digite o caminho completo do arquivo (ex: dados.csv): ")
    if file_name.endswith('.csv'):
        df = pd.read_csv(file_name)
    else:
        df = pd.read_excel(file_name)

df.columns = [str(c).strip() for c in df.columns]
print(f"✅ Base '{file_name}' carregada com sucesso! Total de linhas: {len(df)}")

# ==============================================================================
# 3. FUNÇÕES DE LIMPEZA E PADRONIZAÇÃO DOS DADOS
# ==============================================================================
def clean_binary(x):
    """Converte valores variados de texto/número para formato binário (0 ou 1)."""
    if pd.isna(x): return 0
    s = str(x).strip().upper()
    if s in ['SIM', 'S', 'YES', 'SUCESSO', 'RELEVANTE', 'TRUE', '1']: return 1
    if s == '-': return 0
    try:
        val = float(x)
        if val >= 1: return 1
    except:
        pass
    return 0

def clean_numeric(x):
    """Garante que variáveis de contagem sejam tratadas como números flutuantes."""
    if pd.isna(x): return 0
    if str(x).strip() == '-': return 0
    try:
        return float(x)
    except:
        return 0

# Definição das colunas-chave para as análises
tech_cols = [
    'sigilo_telefonico', 'sigilo_telematico', 'sigilo_bancario', 
    'analise_geolocalizacao', 'interceptacao_telefonica', 
    'cruzamentos_dados', 'softwares_analise'
]
trad_cols = [
    'preservacao_local_crime', 'laudo_pericial_local', 'laudo_cadaverico', 'coleta_vestigios',
    'testemunhas_ouvidas', 'testemunha_ocular_sn', 'familiares_ouvidos_sn', 'oitiva_suspeitos_sn',
    'laudos_complementares', 'diligencias_patrimoniais', 'bloqueio_ativos', 'vigilancias',
    'busca_apreensao', 'prisao_temporaria', 'prisao_preventiva', 'apoio_especializado',
    'cooperacao_externa', 'balistica', 'papiloscopico', 'dna', 'toxicologico', 'imagens_câmera'
]

# Aplicando as limpezas
for col in tech_cols + trad_cols + ['elucidado_sn', 'unid_especializada']:
    if col in df.columns:
        if col == 'testemunhas_ouvidas':
            df[col] = df[col].apply(clean_numeric)
        else:
            df[col] = df[col].apply(clean_binary)

# Criação das variáveis de Volume de Esforço
existing_tech = [c for c in tech_cols if c in df.columns]
existing_all = [c for c in (tech_cols + trad_cols) if c in df.columns]
df['total_atos'] = df[existing_all].sum(axis=1)
df['total_atos_tech'] = df[existing_tech].sum(axis=1)

# Tratamento do Tempo (Removendo o outlier de 4000 dias para não enviesar os modelos)
df['prazo_inquerito'] = pd.to_numeric(df['prazo_inquerito'], errors='coerce')
df_clean = df[(df['prazo_inquerito'] <= 4000) | (df['prazo_inquerito'].isna())].copy()


# ==============================================================================
# 4. ANÁLISE I: RANDOM FOREST (O PESO REAL DAS PROVAS)
# ==============================================================================
print("\n⚙️ Processando Análise I: Random Forest...")
rf_predictors = {
    'prisao_preventiva': 'Prisão Preventiva', 'busca_apreensao': 'Busca e Apreensão',
    'cruzamentos_dados': 'Cruzamento de Dados', 'testemunhas_ouvidas': 'Volume Testemunhas',
    'preservacao_local_crime': 'Preservação de Local', 'balistica': 'Microbalística',
    'testemunha_ocular_sn': 'Testemunha Ocular', 'sigilo_telematico': 'Quebra Telemática',
    'analise_geolocalizacao': 'Geolocalização', 'imagens_câmera': 'Câmeras',
    'softwares_analise': 'Extração Celular', 'sigilo_telefonico': 'Quebra Telefônica',
    'coleta_vestigios': 'Coleta de Vestígios'
}
valid_rf = [p for p in rf_predictors.keys() if p in df_clean.columns]
df_rf = df_clean.dropna(subset=valid_rf + ['elucidado_sn'])

rf = RandomForestClassifier(n_estimators=1000, random_state=42, max_depth=5, class_weight='balanced')
rf.fit(df_rf[valid_rf], df_rf['elucidado_sn'])

df_imp = pd.DataFrame({'Variável': [rf_predictors[col] for col in valid_rf], 'Importância (%)': rf.feature_importances_ * 100})
df_imp = df_imp.sort_values(by='Importância (%)', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importância (%)', y='Variável', data=df_imp, palette='viridis')
plt.title('Importância Preditiva das Diligências (Random Forest)', fontweight='bold')
sns.despine(); plt.tight_layout(); plt.show()


# ==============================================================================
# 5. ANÁLISE II: DECISION TREE (O FLUXOGRAMA DA ELUCIDAÇÃO)
# ==============================================================================
print("⚙️ Processando Análise II: Árvore de Decisão...")
dtree = DecisionTreeClassifier(max_depth=3, random_state=42, class_weight='balanced')
dtree.fit(df_rf[valid_rf], df_rf['elucidado_sn'])

plt.figure(figsize=(16, 8))
plot_tree(dtree, feature_names=[rf_predictors[col] for col in valid_rf], class_names=['Arquivado', 'Indiciado'], filled=True, rounded=True, fontsize=10, proportion=True)
plt.title('Caminhos Lógicos para a Autoria (Decision Tree)', fontweight='bold', fontsize=14)
plt.show()


# ==============================================================================
# 6. ANÁLISE III: NAIVE BAYES (PERFIL PROBABILÍSTICO)
# ==============================================================================
print("⚙️ Processando Análise III: Naive Bayes...")
nb_preds = ['testemunha_ocular_sn', 'preservacao_local_crime', 'balistica', 'cruzamentos_dados', 'sigilo_telematico', 'softwares_analise', 'busca_apreensao', 'prisao_preventiva']
df_nb = df_clean.dropna(subset=nb_preds + ['elucidado_sn'])

nb = BernoulliNB(alpha=1.0)
nb.fit(df_nb[nb_preds], df_nb['elucidado_sn'])

cenarios = [
    {'nome': 'Ausência de Diligências Analisadas', 'valores': [0,0,0,0,0,0,0,0]},
    {'nome': 'Testemunha Ocular + Local', 'valores': [1,1,0,0,0,0,0,0]},
    {'nome': 'Microbalística + Extração Software', 'valores': [0,0,1,0,0,1,0,0]},
    {'nome': 'Busca + Telemática + Cruzamento', 'valores': [0,0,0,1,1,0,1,0]},
    {'nome': 'Combo Completo (Tradicional + Tech)', 'valores': [1,1,1,1,1,1,1,1]}
]

resultados_nb = [{'Cenário': s['nome'], 'Probabilidade (%)': nb.predict_proba([s['valores']])[0][1] * 100} for s in cenarios]
df_res_nb = pd.DataFrame(resultados_nb)

plt.figure(figsize=(10, 5))
ax = sns.barplot(x='Probabilidade (%)', y='Cenário', data=df_res_nb, palette='coolwarm_r')
for p in ax.patches: ax.text(p.get_width()+1, p.get_y()+p.get_height()/2, f"{p.get_width():.1f}%", va='center', fontweight='bold')
plt.title('Probabilidade de Indiciamento por Cenário Probatório (Naive Bayes)', fontweight='bold')
plt.xlim(0, 105); sns.despine(); plt.tight_layout(); plt.show()


# ==============================================================================
# 7. ANÁLISE IV: REGRESSÃO LINEAR (O CUSTO TEMPORAL)
# ==============================================================================
print("⚙️ Processando Análise IV: Regressão Linear Múltipla...")
lr_preds = ['testemunhas_ouvidas', 'busca_apreensao', 'prisao_preventiva', 'cruzamentos_dados', 'softwares_analise', 'sigilo_telematico', 'balistica']
df_lr = df_clean.dropna(subset=lr_preds + ['prazo_inquerito'])

lr = LinearRegression()
lr.fit(df_lr[lr_preds], df_lr['prazo_inquerito'])

df_coef = pd.DataFrame({'Medida': lr_preds, 'Impacto em Dias': lr.coef_}).sort_values(by='Impacto em Dias')
cores = ['#27AE60' if x < 0 else '#C0392B' for x in df_coef['Impacto em Dias']]

plt.figure(figsize=(10, 5))
ax = sns.barplot(x='Impacto em Dias', y='Medida', data=df_coef, palette=cores)
plt.axvline(0, color='black')
plt.title('Impacto Estimado no Prazo Final do Inquérito (Regressão Linear)', fontweight='bold')
sns.despine(); plt.tight_layout(); plt.show()


# ==============================================================================
# 8. ANÁLISE V: REGRESSÃO LOGÍSTICA (A BARREIRA DE ENERGIA)
# ==============================================================================
print("⚙️ Processando Análise V: Regressão Logística...")
df_log = df_clean.dropna(subset=['total_atos', 'elucidado_sn'])
log_reg = LogisticRegression()
log_reg.fit(df_log[['total_atos']], df_log['elucidado_sn'])

threshold = -log_reg.intercept_[0] / log_reg.coef_[0][0]
x_vals = np.linspace(df_log['total_atos'].min(), df_log['total_atos'].max(), 300).reshape(-1, 1)
y_probs = log_reg.predict_proba(x_vals)[:, 1]

plt.figure(figsize=(10, 6))
plt.plot(x_vals, y_probs * 100, color='#2C3E50', linewidth=3)
plt.axvspan(0, threshold, color='#E74C3C', alpha=0.15, label='Zona de Impunidade')
plt.axvspan(threshold, df_log['total_atos'].max() + 2, color='#27AE60', alpha=0.15, label='Zona de Elucidação')
plt.axvline(threshold, color='#34495E', linestyle='--')
plt.title('A "Barreira de Energia" da Investigação Criminal', fontweight='bold')
plt.xlabel('Volume Total de Atos'); plt.ylabel('Probabilidade Matemática (%)')
plt.legend(); sns.despine(); plt.tight_layout(); plt.show()


# ==============================================================================
# 9. ANÁLISE VI: DHPP vs DELEGACIAS TERRITORIAIS
# ==============================================================================
print("⚙️ Processando Análise VI: Especialização Institucional...")
stats_dhpp = df_clean.groupby('unid_especializada').agg(n=('total_atos', 'count'), media_total=('total_atos', 'mean'), media_tech=('total_atos_tech', 'mean')).reset_index()

# Prepara dados para barras agrupadas
data_plot = [
    {'Categoria': 'Total de Atos', 'Grupo': 'DHPP', 'Média': stats_dhpp[stats_dhpp['unid_especializada']==1]['media_total'].values[0]},
    {'Categoria': 'Total de Atos', 'Grupo': 'Territorial', 'Média': stats_dhpp[stats_dhpp['unid_especializada']==0]['media_total'].values[0]},
    {'Categoria': 'Atos Tech', 'Grupo': 'DHPP', 'Média': stats_dhpp[stats_dhpp['unid_especializada']==1]['media_tech'].values[0]},
    {'Categoria': 'Atos Tech', 'Grupo': 'Territorial', 'Média': stats_dhpp[stats_dhpp['unid_especializada']==0]['media_tech'].values[0]}
]
df_plot_dhpp = pd.DataFrame(data_plot)

plt.figure(figsize=(9, 6))
sns.barplot(x='Categoria', y='Média', hue='Grupo', data=df_plot_dhpp, palette=['#27AE60', '#C0392B'])
plt.title('Volume Operacional vs. Aderência Tecnológica (DHPP x Territorial)', fontweight='bold')
sns.despine(); plt.tight_layout(); plt.show()


# ==============================================================================
# 10. ANÁLISE VII: TOP COMBOS (CORRELAÇÃO DE PEARSON)
# ==============================================================================
print("⚙️ Processando Análise VII: Matriz de Correlação (Combos Investigativos)...")
corr_vars = ['busca_apreensao', 'prisao_preventiva', 'cruzamentos_dados', 'softwares_analise', 'sigilo_telematico', 'analise_geolocalizacao', 'sigilo_telefonico', 'balistica']
corr_matrix = df_clean[corr_vars].corr()

# Extrai os pares mais fortes
mask = np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
corr_pairs = corr_matrix.where(mask).stack().reset_index()
corr_pairs.columns = ['Medida A', 'Medida B', 'Correlação (r)']
corr_pairs = corr_pairs.sort_values(by='Correlação (r)', ascending=False).head(5)

print("\n🔥 TOP 5 COMBOS INVESTIGATIVOS NA PCRN:")
print(corr_pairs.to_string(index=False))

print("\n🚀 ANÁLISES CONCLUÍDAS COM SUCESSO! Todos os gráficos foram gerados.")
