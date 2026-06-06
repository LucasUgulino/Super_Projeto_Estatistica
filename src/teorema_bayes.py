def prever_bayes(cliente, prior, likelihood):
    classes = prior.keys()
    prob_posteriori = {}
    
    for c in classes:
        prob = prior[c]
        
        for feature, valor in cliente.items():
            if feature in likelihood and valor in likelihood[feature][c]:
                prob *= likelihood[feature][c][valor]
            else:
                prob *= 1e-6 # Suavização simples para valores não vistos
                
        prob_posteriori[c] = prob
        
    return max(prob_posteriori, key=prob_posteriori.get)
