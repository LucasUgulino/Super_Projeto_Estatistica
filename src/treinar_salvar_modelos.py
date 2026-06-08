import pandas as pd
import numpy as np
import pickle
import json
import os
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

def treinar_e_salvar():
    # Caminhos relativos
    path_data = "data/processed/dados_limpos.parquet"
    path_models_dir = "models"
    
    if not os.path.exists(path_data):
        path_data = "../data/processed/dados_limpos.parquet"
        path_models_dir = "../models"
        
    os.makedirs(path_models_dir, exist_ok=True)
    
    # Carrega dados
    df = pd.read_parquet(path_data)
    df = df.drop_duplicates().reset_index(drop=True)
    
    # 0 = não assinou, 1 = assinou
    df['y'] = df['y'].map({'no': 0, 'yes': 1, 0: 0, 1: 1}).astype(int)
    
    # ==========================================
    # 1. CÁLCULO DAS PROBABILIDADES DE BAYES (MANUAL COM LAPLACE)
    # ==========================================
    features_bayes = [
        'job', 'marital', 'education', 'default', 'housing', 'loan', 
        'contact', 'month', 'day_of_week', 'poutcome', 'faixa_etaria', 
        'faixa_campaign', 'contatado_antes'
    ]
    df_bayes = df[features_bayes + ['y']].copy()
    
    # P(C) - Prior
    counts_y = df_bayes['y'].value_counts()
    total_y = len(df_bayes)
    prior = {str(k): float(v / total_y) for k, v in counts_y.to_dict().items()}
    
    likelihood = {}
    for col in features_bayes:
        likelihood[col] = {}
        # Lista de categorias unicas na feature
        categorias_unicas = df_bayes[col].dropna().unique().tolist()
        v_i = len(categorias_unicas) # cardinalidade |V_i|
        
        for classe in [0, 1]:
            c_str = str(classe)
            likelihood[col][c_str] = {}
            
            # Dados filtrados pela classe
            df_classe = df_bayes[df_bayes['y'] == classe]
            count_classe = len(df_classe)
            
            # Contagem empírica de cada categoria para esta classe
            counts_cat = df_classe[col].value_counts().to_dict()
            
            # Calcula probabilidade de Laplace para cada categoria
            for cat in categorias_unicas:
                count_cat_classe = counts_cat.get(cat, 0)
                prob_laplace = (count_cat_classe + 1) / (count_classe + v_i)
                likelihood[col][c_str][str(cat)] = float(prob_laplace)
                
            # Probabilidade default para categorias não vistas na classe
            prob_default = 1 / (count_classe + v_i)
            likelihood[col][c_str]["__default__"] = float(prob_default)
            
    bayes_dados = {
        "prior": prior,
        "likelihood": likelihood
    }
    
    with open(os.path.join(path_models_dir, "bayes_dados.json"), "w") as f:
        json.dump(bayes_dados, f, indent=4)
    print("Dados do Bayes manual salvos em JSON.")
    
    # ==========================================
    # 2. TREINAMENTO DOS MODELOS DE ML (SCIKIT-LEARN)
    # ==========================================
    X = df.drop(columns=['y'])
    y = df['y']
    
    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)
    
    # Divisão treino/teste
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Salvar colunas pós-encoding
    colunas_treino = X_train.columns.tolist()
    with open(os.path.join(path_models_dir, "colunas_treino.json"), "w") as f:
        json.dump(colunas_treino, f)
        
    # Escalonamento
    colunas_numericas = df.select_dtypes(include=['number']).drop(columns=['y']).columns.tolist()
    
    X_train_log = X_train.copy()
    scaler = StandardScaler()
    
    # Treina o scaler
    X_train_log[colunas_numericas] = scaler.fit_transform(X_train_log[colunas_numericas])
    
    # Salva o scaler
    with open(os.path.join(path_models_dir, "scaler.pkl"), "wb") as f:
        pickle.dump(scaler, f)
        
    # Treina Regressão Logística
    modelo_logistico = LogisticRegression(max_iter=1000, class_weight='balanced', random_state=42)
    modelo_logistico.fit(X_train_log, y_train)
    
    with open(os.path.join(path_models_dir, "modelo_logistico.pkl"), "wb") as f:
        pickle.dump(modelo_logistico, f)
        
    # Treina Árvore de Decisão
    modelo_arvore = DecisionTreeClassifier(max_depth=5, min_samples_leaf=50, class_weight='balanced', random_state=42)
    modelo_arvore.fit(X_train, y_train)
    
    with open(os.path.join(path_models_dir, "modelo_arvore.pkl"), "wb") as f:
        pickle.dump(modelo_arvore, f)
        
    print("Modelos (Regressão Logística, Árvore de Decisão e Scaler) salvos com sucesso em formato pickle.")

if __name__ == "__main__":
    treinar_e_salvar()
