def calcular_probabilidades_posteriori(cliente, prior, likelihood):
    """
    Calcula as probabilidades a posteriori normalizadas P(C|X) para cada classe C.
    Aplica a probabilidade default (__default__) caso a categoria não tenha sido vista no treino.
    """
    classes = list(prior.keys())
    scores_posteriori = {}
    
    for c in classes:
        # P(C) - Prior
        prob_c = prior[c]
        
        # P(X|C) = Produtorio de P(Xi|C)
        prob_x_dado_c = 1.0
        for feature, valor in cliente.items():
            if feature in likelihood:
                # Se o valor existe no treino daquela classe, usa; senão usa o valor default de Laplace
                if valor in likelihood[feature][c]:
                    prob_x_dado_c *= likelihood[feature][c][valor]
                else:
                    # Suavização de Laplace para valores não vistos
                    prob_x_dado_c *= likelihood[feature][c].get("__default__", 1e-6)
                    
        scores_posteriori[c] = prob_x_dado_c * prob_c
        
    # Normalização P(C|X) = P(X|C)*P(C) / Soma(P(X|C_k)*P(C_k))
    soma_evidencias = sum(scores_posteriori.values())
    
    prob_posteriori = {}
    if soma_evidencias > 0:
        for c in classes:
            prob_posteriori[c] = scores_posteriori[c] / soma_evidencias
    else:
        # Fallback uniforme se todas derem zero devido a underflow
        for c in classes:
            prob_posteriori[c] = 1.0 / len(classes)
            
    return prob_posteriori

def prever_bayes(cliente, prior, likelihood):
    """
    Preve a classe do cliente ('Compra' ou 'Não compra') e a probabilidade de compra (classe '1' ou 1).
    """
    prob_posteriori = calcular_probabilidades_posteriori(cliente, prior, likelihood)
    
    # Mapeamento de chaves (pode vir como string ou int do JSON)
    classe_yes_key = '1' if '1' in prob_posteriori else 1
    classe_no_key = '0' if '0' in prob_posteriori else 0
    
    prob_yes = prob_posteriori.get(classe_yes_key, 0.0)
    
    decisao = "Compra" if prob_yes > 0.5 else "Não compra"
    return decisao, prob_yes
