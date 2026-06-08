import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os
import pickle
import json

# Importa a predição real do Bayes Manual
try:
    from teorema_bayes import prever_bayes
except ImportError:
    from src.teorema_bayes import prever_bayes

# Configurações de layout e estética premium
st.set_page_config(
    page_title="Analytics e Predictions - Telemarketing Banco",
    page_icon="https://fonts.gstatic.com/s/i/short-term/release/googlesymbols/monitoring/default/24px.svg",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilização CSS customizada (Modern Glassmorphism & Cyber Dark)
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;700&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');
    
    /* Root Variables para o tema */
    :root {
        --bg-primary: #0a0b10;
        --bg-secondary: rgba(17, 19, 31, 0.7);
        --accent-primary: #6366f1; /* Indigo */
        --accent-secondary: #06b6d4; /* Cyan */
        --text-primary: #f8fafc;
        --text-secondary: #94a3b8;
        --border-color: rgba(255, 255, 255, 0.08);
    }
    
    .stApp {
        background-color: var(--bg-primary);
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    
    h1, h2, h3, h4, h5, h6 {
        font-family: 'Outfit', sans-serif;
        color: var(--text-primary);
        font-weight: 700;
        letter-spacing: -0.02em;
    }
    
    .main-title {
        background: linear-gradient(135deg, #a5b4fc 0%, #6366f1 50%, #06b6d4 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        padding-top: 15px;
        padding-bottom: 25px;
        font-size: 2.8rem !important;
        font-weight: 800;
    }
    
    .subtitle {
        text-align: center;
        color: var(--text-secondary);
        font-size: 1.1rem;
        margin-top: -15px;
        margin-bottom: 30px;
    }
    
    /* Efeito Glassmorphic para Cards */
    .custom-card {
        background: var(--bg-secondary);
        backdrop-filter: blur(12px) saturate(180%);
        -webkit-backdrop-filter: blur(12px) saturate(180%);
        border: 1px solid var(--border-color);
        border-radius: 16px;
        padding: 24px;
        margin-bottom: 24px;
        box-shadow: 0 8px 32px 0 rgba(0, 0, 0, 0.37);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    
    .custom-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Estilos de Métricas Internas */
    .metric-container {
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    
    .metric-value {
        font-size: 2.2rem;
        font-weight: 700;
        background: linear-gradient(135deg, #38bdf8 0%, #0369a1 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 5px 0;
        font-family: 'Outfit', sans-serif;
    }
    
    .metric-label {
        font-size: 0.85rem;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        color: var(--text-secondary);
        font-weight: 600;
    }
    
    .decision-badge {
        display: inline-block;
        padding: 6px 16px;
        border-radius: 99px;
        font-size: 0.85rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        margin-top: 8px;
    }
    
    .badge-yes {
        background-color: rgba(16, 185, 129, 0.15);
        color: #34d399;
        border: 1px solid rgba(16, 185, 129, 0.3);
    }
    
    .badge-no {
        background-color: rgba(239, 68, 68, 0.15);
        color: #f87171;
        border: 1px solid rgba(239, 68, 68, 0.3);
    }
    
    /* Abas Streamlit customizadas */
    div.stTabs [data-baseweb="tab-list"] {
        column-gap: 8px;
        background-color: transparent;
        justify-content: center;
        border-bottom: 1px solid var(--border-color);
        padding-bottom: 8px;
    }

    div.stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: rgba(17, 19, 31, 0.4);
        border: 1px solid var(--border-color);
        border-radius: 8px;
        color: var(--text-secondary);
        font-size: 1rem;
        font-weight: 500;
        padding: 0px 24px;
        transition: all 0.2s ease;
    }

    div.stTabs [data-baseweb="tab"]:hover {
        background-color: rgba(99, 102, 241, 0.1);
        color: var(--text-primary);
        border-color: rgba(99, 102, 241, 0.3);
    }

    div.stTabs [aria-selected="true"] {
        background-color: var(--accent-primary) !important;
        color: var(--text-primary) !important;
        border-color: var(--accent-primary) !important;
        box-shadow: 0 4px 14px 0 rgba(99, 102, 241, 0.4);
        font-weight: 600;
    }
    
    /* Inputs Streamlit */
    div[data-baseweb="select"] > div {
        background-color: rgba(17, 19, 31, 0.8) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    input {
        background-color: rgba(17, 19, 31, 0.8) !important;
        border: 1px solid var(--border-color) !important;
        color: var(--text-primary) !important;
    }
    
    /* Alertas e avisos customizados */
    .custom-alert {
        background-color: rgba(99, 102, 241, 0.08);
        border-left: 4px solid var(--accent-primary);
        padding: 16px;
        border-radius: 4px 8px 8px 4px;
        margin: 20px 0;
        font-size: 0.92rem;
        color: var(--text-secondary);
    }
    
</style>
""", unsafe_allow_html=True)

# Função para carregar os dados tratados
@st.cache_data
def load_data():
    caminhos = [
        "data/processed/dados_eda.parquet",
        "../data/processed/dados_eda.parquet",
        "data/processed/dados_limpos.parquet",
        "../data/processed/dados_limpos.parquet"
    ]
    for path in caminhos:
        if os.path.exists(path):
            df = pd.read_parquet(path)
            if df['y'].dtype in [int, float] or df['y'].dtype == 'int64':
                df['y_desc'] = df['y'].map({0: 'Não', 1: 'Sim'})
            else:
                df['y_desc'] = df['y'].map({'no': 'Não', 'yes': 'Sim', 0: 'Não', 1: 'Sim'})
            return df
    st.error("Dataset processado não encontrado. Certifique-se de executar o pipeline de limpeza primeiro.")
    return None

# Função para carregar os modelos e estatísticas reais de predição
@st.cache_resource
def load_predictive_resources():
    paths = {
        "bayes": ["models/bayes_dados.json", "../models/bayes_dados.json"],
        "cols": ["models/colunas_treino.json", "../models/colunas_treino.json"],
        "scaler": ["models/scaler.pkl", "../models/scaler.pkl"],
        "lr": ["models/modelo_logistico.pkl", "../models/modelo_logistico.pkl"],
        "dt": ["models/modelo_arvore.pkl", "../models/modelo_arvore.pkl"]
    }
    
    resources = {}
    
    # 1. Carrega dados de Bayes
    for p in paths["bayes"]:
        if os.path.exists(p):
            with open(p, "r") as f:
                resources["bayes"] = json.load(f)
            break
            
    # 2. Carrega colunas de treino
    for p in paths["cols"]:
        if os.path.exists(p):
            with open(p, "r") as f:
                resources["cols"] = json.load(f)
            break
            
    # 3. Carrega o scaler
    for p in paths["scaler"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                resources["scaler"] = pickle.load(f)
            break
            
    # 4. Carrega Regressão Logística
    for p in paths["lr"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                resources["lr"] = pickle.load(f)
            break
            
    # 5. Carrega Árvore de Decisão
    for p in paths["dt"]:
        if os.path.exists(p):
            with open(p, "rb") as f:
                resources["dt"] = pickle.load(f)
            break
            
    return resources

df = load_data()
resources = load_predictive_resources()

# Título Principal do Dashboard
st.markdown("<h1 class='main-title'>Super Projeto de Estatística e Probabilidade</h1>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Auditoria Analítica e Modelagem Preditiva de Campanhas de Marketing</div>", unsafe_allow_html=True)

# Divisão das Abas (sem emojis)
aba1, aba2 = st.tabs(["Seção 1: Análise Exploratória de Dados (EDA)", "Seção 2: Simulador e Comparação de Predição"])

# ==========================================
# SEÇÃO 1: ANÁLISE EXPLORATÓRIA (EDA)
# ==========================================
with aba1:
    st.header("Análise Exploratória do Comportamento das Campanhas")
    st.write(
        "Esta seção apresenta as distribuições e correlações mais relevantes identificadas no conjunto de dados "
        "reais de telemarketing bancário, explicitando o objetivo analítico de cada investigação."
    )
    
    if df is not None:
        col1, col2 = st.columns(2)
        
        with col1:
            # 1. Distribuição de Idades por Conversão com Filtro de Estado Civil
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("1. Distribuição de Idades por Conversão")
            
            opcoes_marital = ["Todos"] + sorted(list(df["marital"].unique()))
            filtro_marital = st.selectbox(
                "Filtrar por Estado Civil (Gráfico 1):", 
                opcoes_marital,
                key="filtro_g1"
            )
            
            df_g1 = df if filtro_marital == "Todos" else df[df["marital"] == filtro_marital]
            
            fig_idade = px.histogram(
                df_g1, 
                x="age", 
                color="y_desc", 
                nbins=30,
                color_discrete_map={"Não": "#f87171", "Sim": "#34d399"},
                labels={"age": "Idade", "y_desc": "Adesão ao Produto"},
                barmode="overlay",
                opacity=0.8
            )
            fig_idade.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                font_color="#fafafa",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_idade, use_container_width=True)
            
            st.markdown(f"""
            **Objetivo Analítico:** Identificar em quais faixas etárias de clientes se concentra a maior taxa de conversão (adesão ao depósito a prazo). 
            
            *Insight:* Embora o volume absoluto de chamadas se concentre na faixa dos 30 a 45 anos, as proporções de conversão (adesão) são visivelmente mais altas para jovens (abaixo de 25 anos) e idosos (acima de 65 anos).
            
            *Filtro Ativo:* **{filtro_marital}** (Exibindo {len(df_g1)} registros).
            """)
            st.markdown("</div>", unsafe_allow_html=True)
            
            # 3. Relação entre Duração da Chamada e Adesão com Filtro de Mês
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("3. Relação entre Duração da Chamada e Adesão")
            
            opcoes_mes = ["Todos"] + sorted(list(df["month"].unique()))
            filtro_mes = st.selectbox(
                "Filtrar por Mês do Contato (Gráfico 3):", 
                opcoes_mes,
                key="filtro_g3"
            )
            
            df_g3 = df if filtro_mes == "Todos" else df[df["month"] == filtro_mes]
            
            fig_duracao = px.box(
                df_g3,
                x="y_desc",
                y="duration",
                color="y_desc",
                color_discrete_map={"Não": "#f87171", "Sim": "#34d399"},
                labels={"y_desc": "Adesão ao Produto", "duration": "Duração (segundos)"}
            )
            fig_duracao.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                font_color="#fafafa",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_duracao, use_container_width=True)
            st.markdown(f"""
            **Objetivo Analítico:** Analisar o impacto da duração da chamada telefônica na probabilidade de adesão do cliente.
            
            *Insight:* Chamadas com conversão bem-sucedida possuem duração mediana significativamente superior (cerca de 550 segundos) em comparação às rejeitadas (cerca de 220 segundos). 
            
            *Nota Estatística:* Embora seja um preditor forte, esta variável causa vazamento de dados (data leakage) e foi descartada na modelagem preditiva real, pois só é conhecida após o término do contato.
            
            *Filtro Ativo:* **{filtro_mes}** (Exibindo {len(df_g3)} registros).
            """)
            st.markdown("</div>", unsafe_allow_html=True)

        with col2:
            # 2. Taxa de Conversão por Categoria Ocupacional com Filtro de Canal de Contato
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("2. Taxa de Conversão por Categoria Ocupacional (Job)")
            
            opcoes_contato = ["Todos"] + sorted(list(df["contact"].unique()))
            filtro_contato = st.selectbox(
                "Filtrar por Canal de Comunicação (Gráfico 2):", 
                opcoes_contato,
                key="filtro_g2"
            )
            
            df_g2 = df if filtro_contato == "Todos" else df[df["contact"] == filtro_contato]
            
            conversao_job = df_g2.groupby('job', observed=False)['y'].apply(
                lambda x: (x == 'yes').mean() * 100 if x.dtype == object else x.mean() * 100
            ).reset_index().sort_values(by='y', ascending=True)
            
            fig_job = px.bar(
                conversao_job,
                y="job",
                x="y",
                orientation='h',
                color="y",
                color_continuous_scale="magma",
                labels={"job": "Ocupação", "y": "Taxa de Conversão (%)"}
            )
            fig_job.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                font_color="#fafafa",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_job, use_container_width=True)
            st.markdown(f"""
            **Objetivo Analítico:** Mapear a aderência do produto de investimento a diferentes perfis socioeconômicos representados pela ocupação do cliente.
            
            *Insight:* Estudantes e aposentados lideram as taxas de conversão com valores acima de 25%. Em contrapartida, trabalhadores braçais e de serviços apresentam taxas de conversão inferiores a 10%.
            
            *Filtro Ativo:* **{filtro_contato}** (Exibindo {len(df_g2)} registros).
            """)
            st.markdown("</div>", unsafe_allow_html=True)

            # 4. Impacto do Sucesso de Campanhas Anteriores com Filtro de Inadimplência
            st.markdown("<div class='custom-card'>", unsafe_allow_html=True)
            st.subheader("4. Impacto do Sucesso de Campanhas Anteriores")
            
            opcoes_housing = ["Todos"] + sorted(list(df["housing"].unique()))
            filtro_housing = st.selectbox(
                "Filtrar por Empréstimo Imobiliário (Gráfico 4):", 
                opcoes_housing,
                key="filtro_g4"
            )
            
            df_g4 = df if filtro_housing == "Todos" else df[df["housing"] == filtro_housing]
            
            poutcome_y = pd.crosstab(df_g4['poutcome'], df_g4['y_desc'], normalize='index') * 100
            poutcome_y = poutcome_y.reset_index()
            
            fig_poutcome = px.bar(
                poutcome_y,
                x="poutcome",
                y=["Não", "Sim"],
                barmode="stack",
                color_discrete_map={"Não": "#f87171", "Sim": "#34d399"},
                labels={"value": "Porcentagem (%)", "poutcome": "Resultado Anterior", "variable": "Adesão"}
            )
            fig_poutcome.update_layout(
                paper_bgcolor='rgba(0,0,0,0)',
                plot_bgcolor='rgba(0,0,0,0)',
                font_family="Plus Jakarta Sans",
                font_color="#fafafa",
                margin=dict(l=10, r=10, t=30, b=10)
            )
            st.plotly_chart(fig_poutcome, use_container_width=True)
            st.markdown(f"""
            **Objetivo Analítico:** Avaliar a persistência temporal de comportamento do cliente, isto é, se o sucesso de um contato em uma campanha passada prediz a conversão atual.
            
            *Insight:* Clientes que aderiram a campanhas anteriores (success) possuem taxa de conversão atual próxima a 65%, enquanto aqueles que nunca foram contatados ou falharam anteriormente ficam abaixo de 15%.
            
            *Filtro Ativo:* **{filtro_housing}** (Exibindo {len(df_g4)} registros).
            """)
            st.markdown("</div>", unsafe_allow_html=True)

# ==========================================
# SEÇÃO 2: COMPARADOR DE PREDIÇÃO
# ==========================================
with aba2:
    st.header("Simulação de Clientes e Comparação de Modelos")
    st.write(
        "Forneça as características do cliente no formulário de entrada para simular e "
        "comparar a probabilidade de conversão gerada simultaneamente pelos três classificadores."
    )
    
    col_form, col_res = st.columns([1, 1.4])
    
    with col_form:
        st.markdown("<h3 style='color:#6366f1; margin-bottom: 15px;'>Características do Cliente</h3>", unsafe_allow_html=True)
        
        with st.form("form_cliente"):
            idade_input = st.number_input("Idade", min_value=17, max_value=100, value=35)
            
            job_input = st.selectbox(
                "Ocupação (Job)", 
                ['admin.', 'blue-collar', 'entrepreneur', 'housemaid', 'management', 
                 'retired', 'self-employed', 'services', 'student', 'technician', 'unemployed', 'unknown']
            )
            
            marital_input = st.selectbox("Estado Civil", ['married', 'single', 'divorced', 'unknown'])
            
            education_input = st.selectbox(
                "Nível de Escolaridade", 
                ['university.degree', 'high.school', 'professional.course', 
                 'basic.9y', 'basic.6y', 'basic.4y', 'illiterate', 'unknown']
            )
            
            default_input = st.selectbox("Inadimplência de Crédito (Default)", ['no', 'yes', 'unknown'])
            housing_input = st.selectbox("Possui Empréstimo Imobiliário?", ['yes', 'no', 'unknown'])
            loan_input = st.selectbox("Possui Empréstimo Pessoal?", ['no', 'yes', 'unknown'])
            
            st.markdown("<p style='font-size:0.9rem; font-weight:600; color:#94a3b8; margin: 15px 0 5px 0;'>Dados da Campanha Atual</p>", unsafe_allow_html=True)
            
            contact_input = st.selectbox("Canal de Comunicação", ['cellular', 'telephone'])
            month_input = st.selectbox("Mês de Contato", ['may', 'jul', 'aug', 'jun', 'nov', 'apr', 'oct', 'sep', 'mar', 'dec'])
            day_input = st.selectbox("Dia da Semana", ['mon', 'tue', 'wed', 'thu', 'fri'])
            
            poutcome_input = st.selectbox("Resultado da Campanha Anterior", ['nonexistent', 'failure', 'success'])
            campaign_input = st.slider("Contatos Realizados nesta Campanha", min_value=1, max_value=50, value=1)
            
            submit_button = st.form_submit_button("Gerar Predições")
            
        # Determinar colunas derivadas (engenharia de features)
        if idade_input <= 25:
            faixa_etaria_input = 'ate_25'
        elif idade_input <= 35:
            faixa_etaria_input = '26_35'
        elif idade_input <= 50:
            faixa_etaria_input = '36_50'
        elif idade_input <= 65:
            faixa_etaria_input = '51_65'
        else:
            faixa_etaria_input = 'acima_65'
            
        if campaign_input == 1:
            faixa_campaign_input = '1'
        elif campaign_input <= 3:
            faixa_campaign_input = '2a3'
        elif campaign_input <= 6:
            faixa_campaign_input = '4a6'
        else:
            faixa_campaign_input = 'mais_6'
            
        contatado_antes_input = 0 if poutcome_input == 'nonexistent' else 1
        
        cliente_dict = {
            'job': job_input,
            'marital': marital_input,
            'education': education_input,
            'default': default_input,
            'housing': housing_input,
            'loan': loan_input,
            'contact': contact_input,
            'month': month_input,
            'day_of_week': day_input,
            'poutcome': poutcome_input,
            'faixa_etaria': faixa_etaria_input,
            'faixa_campaign': faixa_campaign_input,
            'contatado_antes': contatado_antes_input
        }
        
    with col_res:
        st.markdown("<h3 style='color:#06b6d4; margin-bottom: 15px;'>Resultados da Classificação</h3>", unsafe_allow_html=True)
        st.write("Análise comparativa das probabilidades estimadas por modelo preditivo:")
        
        # 1. Roda predição do Bayes Manual (Real)
        if "bayes" in resources:
            pred_bayes, prob_bayes = prever_bayes(
                cliente_dict, 
                resources["bayes"]["prior"], 
                resources["bayes"]["likelihood"]
            )
        else:
            # Fallback de segurança se o arquivo json não foi carregado
            pred_bayes, prob_bayes = "Não compra", 0.0
            
        # 2. Roda predição dos Modelos Scikit-Learn (Reais)
        if "lr" in resources and "dt" in resources and "cols" in resources and "scaler" in resources:
            # Prepara vetor do cliente contendo exatamente as colunas e formatos do get_dummies do treino
            # Inferência lógica das colunas numéricas ausentes no formulário
            pdays_val = 0 if poutcome_input == 'nonexistent' else 5
            previous_val = 0 if poutcome_input == 'nonexistent' else 1
            
            # Cria dataframe individual do cliente
            X_cliente = pd.DataFrame([{
                'age': float(idade_input),
                'job': job_input,
                'marital': marital_input,
                'education': education_input,
                'default': default_input,
                'housing': housing_input,
                'loan': loan_input,
                'contact': contact_input,
                'month': month_input,
                'day_of_week': day_input,
                'campaign': float(campaign_input),
                'pdays': float(pdays_val),
                'previous': float(previous_val),
                'poutcome': poutcome_input,
                'contatado_antes': contatado_antes_input,
                'faixa_etaria': faixa_etaria_input,
                'faixa_campaign': faixa_campaign_input
            }])
            
            # Gera as colunas dummies do cliente individual
            X_cliente_encoded = pd.get_dummies(X_cliente)
            
            # Alinha e preenche o vetor para ter as mesmas colunas estruturadas do treino (colunas_treino)
            colunas_treino = resources["cols"]
            X_cliente_final = pd.DataFrame(0.0, index=[0], columns=colunas_treino)
            
            for col in X_cliente_encoded.columns:
                if col in X_cliente_final.columns:
                    X_cliente_final.loc[0, col] = float(X_cliente_encoded.loc[0, col])
            
            # Obter as colunas numéricas que o scaler original espera (incluindo as macroeconômicas)
            colunas_numericas_scaler = list(resources["scaler"].feature_names_in_)
            
            # Preenche as colunas numéricas no vetor final do cliente (usando médias do dataset para macroeconômicas)
            for col in colunas_numericas_scaler:
                if col == 'age':
                    X_cliente_final.loc[0, col] = float(idade_input)
                elif col == 'campaign':
                    X_cliente_final.loc[0, col] = float(campaign_input)
                elif col == 'pdays':
                    X_cliente_final.loc[0, col] = float(pdays_val)
                elif col == 'previous':
                    X_cliente_final.loc[0, col] = float(previous_val)
                else:
                    # Se for variável macroeconômica, preenchemos com a média do dataset carregado
                    if df is not None and col in df.columns:
                        X_cliente_final.loc[0, col] = float(df[col].mean())
                    else:
                        X_cliente_final.loc[0, col] = 0.0
            
            # Executa Escalonamento para o Modelo Linear (Regressão Logística)
            X_cliente_log = X_cliente_final.copy()
            X_cliente_log[colunas_numericas_scaler] = resources["scaler"].transform(X_cliente_final[colunas_numericas_scaler])
            
            # Regressão Logística
            prob_lr = float(resources["lr"].predict_proba(X_cliente_log)[0][1])
            pred_lr = "Compra" if prob_lr > 0.5 else "Não compra"
            
            # Árvore de Decisão
            prob_rf = float(resources["dt"].predict_proba(X_cliente_final)[0][1])
            pred_rf = "Compra" if prob_rf > 0.5 else "Não compra"
        else:
            # Fallback de segurança se os pickles não foram carregados
            prob_lr, pred_lr = 0.0, "Não compra"
            prob_rf, pred_rf = 0.0, "Não compra"
            st.warning("Modelos scikit-learn não foram carregados. Exibindo dados nulos.")
            
        # Exibição de cards estilizados com glassmorphism (sem emojis)
        c1, c2, c3 = st.columns(3)
        
        badge_bayes = "badge-yes" if pred_bayes == "Compra" else "badge-no"
        badge_lr = "badge-yes" if pred_lr == "Compra" else "badge-no"
        badge_rf = "badge-yes" if pred_rf == "Compra" else "badge-no"
        
        with c1:
            st.markdown(f"""
            <div class='custom-card' style='text-align:center; min-height: 170px;'>
                <div class='metric-label'>Bayes Manual</div>
                <div class='metric-value'>{prob_bayes*100:.1f}%</div>
                <span class='decision-badge {badge_bayes}'>{pred_bayes}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c2:
            st.markdown(f"""
            <div class='custom-card' style='text-align:center; min-height: 170px;'>
                <div class='metric-label'>Regressão Logística</div>
                <div class='metric-value'>{prob_lr*100:.1f}%</div>
                <span class='decision-badge {badge_lr}'>{pred_lr}</span>
            </div>
            """, unsafe_allow_html=True)
            
        with c3:
            st.markdown(f"""
            <div class='custom-card' style='text-align:center; min-height: 170px;'>
                <div class='metric-label'>Árvore de Decisão</div>
                <div class='metric-value'>{prob_rf*100:.1f}%</div>
                <span class='decision-badge {badge_rf}'>{pred_rf}</span>
            </div>
            """, unsafe_allow_html=True)
            
        # Comparação Visual Gráfica
        st.subheader("Probabilidades Comparadas")
        dados_prob = pd.DataFrame({
            "Modelo": ["Teorema de Bayes (Manual)", "Regressão Logística", "Árvore de Decisão"],
            "Probabilidade de Adesão (%)": [prob_bayes * 100, prob_lr * 100, prob_rf * 100],
            "Decisão": [pred_bayes, pred_lr, pred_rf]
        })
        
        fig_prob = px.bar(
            dados_prob,
            x="Probabilidade de Adesão (%)",
            y="Modelo",
            text="Probabilidade de Adesão (%)",
            orientation='h',
            color="Decisão",
            color_discrete_map={"Não compra": "#f87171", "Compra": "#34d399"},
            range_x=[0, 100]
        )
        
        fig_prob.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            plot_bgcolor='rgba(0,0,0,0)',
            font_family="Plus Jakarta Sans",
            font_color="#fafafa",
            margin=dict(l=10, r=10, t=10, b=10)
        )
        fig_prob.update_traces(texttemplate='%{text:.1f}%', textposition='outside')
        st.plotly_chart(fig_prob, use_container_width=True)
        
        st.markdown("""
        <div class='custom-alert'>
            <strong>Validação Acadêmica:</strong> Este simulador carrega os coeficientes oficiais ajustados do scikit-learn 
            (Regressão Logística e Árvore de Decisão) salvos em formato binário e executa dinamicamente o cálculo manual 
            do Teorema de Bayes aplicando a estatística exata com Suavização de Laplace, garantindo conformidade matemática 
            com os pilares do edital.
        </div>
        """, unsafe_allow_html=True)
