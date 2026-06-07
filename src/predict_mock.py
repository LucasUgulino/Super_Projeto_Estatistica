import numpy as np

def predict_bayes_manual(cliente_dict):
    """
    Simula o comportamento de predição do classificador Naive Bayes manual.
    Retorna a classe predita ('Não compra' ou 'Compra') e a probabilidade de conversão.
    """
    # Lógica baseada em correlações reais do dataset Bank Marketing
    score = 0.11  # Taxa de conversão média global (~11%)
    
    # Se foi contatado antes com sucesso, a probabilidade aumenta drasticamente
    if cliente_dict.get('poutcome') == 'success':
        score += 0.50
    elif cliente_dict.get('poutcome') == 'nonexistent':
        score -= 0.02
        
    # Faixa etária
    faixa_etaria = cliente_dict.get('faixa_etaria')
    if faixa_etaria in ['ate_25', 'acima_65']:
        score += 0.15  # Jovens e idosos convertem mais
    elif faixa_etaria == '36_50':
        score -= 0.03
        
    # Trabalho
    job = cliente_dict.get('job')
    if job in ['student', 'retired']:
        score += 0.12
    elif job in ['blue-collar', 'services']:
        score -= 0.04
        
    # Tipo de contato
    if cliente_dict.get('contact') == 'cellular':
        score += 0.05
    else:
        score -= 0.05
        
    # Limitar score entre 0.01 e 0.99
    prob_yes = float(np.clip(score, 0.01, 0.99))
    prob_no = 1.0 - prob_yes
    
    pred = "Compra" if prob_yes > 0.5 else "Não compra"
    
    return pred, prob_yes

def predict_logistic_regression(cliente_dict):
    """
    Simula a predição de um modelo de Regressão Logística.
    """
    # Semelhante ao Bayes, mas com pesos ligeiramente diferentes (comportamento de modelo linear)
    score = 0.10
    
    if cliente_dict.get('poutcome') == 'success':
        score += 0.55
    elif cliente_dict.get('poutcome') == 'failure':
        score += 0.05
        
    faixa_etaria = cliente_dict.get('faixa_etaria')
    if faixa_etaria == 'acima_65':
        score += 0.20
    elif faixa_etaria == 'ate_25':
        score += 0.10
        
    job = cliente_dict.get('job')
    if job == 'student':
        score += 0.15
    elif job == 'blue-collar':
        score -= 0.05
        
    if cliente_dict.get('month') in ['mar', 'sep', 'oct', 'dec']:
        score += 0.10 # Meses com alta taxa de conversão no dataset
        
    prob_yes = float(np.clip(score, 0.01, 0.99))
    pred = "Compra" if prob_yes > 0.5 else "Não compra"
    
    return pred, prob_yes

def predict_random_forest(cliente_dict):
    """
    Simula a predição de um modelo Random Forest (não linear, interações complexas).
    """
    score = 0.12
    
    # Random Forest captura interações não-lineares
    poutcome = cliente_dict.get('poutcome')
    job = cliente_dict.get('job')
    
    if poutcome == 'success':
        if job in ['student', 'retired', 'admin.']:
            score += 0.65
        else:
            score += 0.40
    else:
        if job == 'student' and cliente_dict.get('faixa_etaria') == 'ate_25':
            score += 0.18
        elif job == 'blue-collar':
            score -= 0.06
            
    if cliente_dict.get('default') == 'yes':
        score -= 0.30
        
    prob_yes = float(np.clip(score, 0.01, 0.99))
    pred = "Compra" if prob_yes > 0.5 else "Não compra"
    
    return pred, prob_yes
