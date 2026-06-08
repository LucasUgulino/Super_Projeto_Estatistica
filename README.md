
# Super Projeto de Estatística e Probabilidade

Projeto acadêmico desenvolvido com o dataset **Bank Marketing**, do UCI Machine Learning Repository, para analisar campanhas de telemarketing bancário e prever se um cliente irá aderir a um depósito a prazo.

O trabalho integra tratamento e limpeza de dados, análise exploratória, aplicação manual do Teorema de Bayes, treinamento de algoritmos de classificação e construção de um dashboard interativo em Streamlit.

## Objetivo

Investigar quais características do cliente, do contato e da campanha estão associadas à maior probabilidade de assinatura de um depósito a prazo.

A variável alvo do projeto é `y`, que indica se o cliente aceitou (`yes`/`1`) ou recusou (`no`/`0`) a oferta.

## Dataset

O dataset utilizado foi o **Bank Marketing**, disponibilizado pelo UCI Machine Learning Repository.

- Fonte: [UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)
- Arquivo principal: `bank-additional-full.csv`
- Registros originais: 41.188
- Atributos originais: 20 variáveis de entrada + variável alvo `y`
- Domínio: campanhas de marketing telefônico de uma instituição bancária portuguesa
- Tipo de problema: classificação binária

O arquivo `bank-additional-names.txt` foi usado como dicionário de dados para interpretar as colunas, categorias e referências da base.

## Estrutura do projeto

```text
.
├── data/
│   ├── raw/
│   │   ├── bank-additional-full.csv
│   │   └── bank-additional-names.txt
│   └── processed/
│       ├── dados_limpos.csv
│       ├── dados_limpos.parquet
│       ├── dados_eda.csv
│       └── dados_eda.parquet
├── models/
│   ├── bayes_dados.json
│   ├── colunas_treino.json
│   ├── modelo_arvore.pkl
│   ├── modelo_logistico.pkl
│   └── scaler.pkl
├── notebooks/
│   ├── tratamento_limpeza.ipynb
│   ├── eda.ipynb
│   ├── teorema_bayes.ipynb
│   └── pre_processamento_ml.ipynb
├── src/
│   ├── app.py
│   ├── teorema_bayes.py
│   ├── predict_mock.py
│   └── treinar_salvar_modelos.py
├── requirements.txt
└── README.md
```

## Etapas desenvolvidas

### 1. Tratamento e limpeza dos dados

Foram aplicadas etapas de preparação para melhorar a qualidade da base e deixá-la adequada para análise estatística, Bayes e modelos de classificação.

Principais tratamentos:

- leitura do CSV com separador `;`;
- verificação e remoção de duplicatas;
- tratamento de valores `unknown`;
- conversão da variável alvo `y` para formato binário;
- criação da variável `contatado_antes`, a partir de `pdays`;
- substituição do valor especial `pdays = 999`, que indica ausência de contato anterior;
- criação de faixas para idade (`faixa_etaria`);
- criação de faixas para quantidade de contatos (`faixa_campaign`);
- separação entre dados usados na EDA e dados usados na modelagem.

A variável `duration` foi mantida na base de EDA, mas removida da base principal de modelagem para evitar vazamento de dados, já que a duração da chamada só é conhecida após o contato com o cliente.

### 2. Análise exploratória de dados

A EDA foi construída para identificar padrões relacionados à adesão ao depósito a prazo.

Foram analisados pontos como:

- distribuição de idade por conversão;
- taxa de conversão por ocupação (`job`);
- relação entre duração da chamada e adesão;
- impacto do resultado de campanhas anteriores (`poutcome`);
- desbalanceamento da variável alvo;
- comportamento das variáveis qualitativas e quantitativas.

Alguns insights observados:

- clientes com `poutcome = success` apresentam maior probabilidade de conversão;
- estudantes e aposentados tendem a ter taxas de adesão mais altas;
- clientes jovens e idosos apresentam proporções de conversão relevantes;
- chamadas mais longas aparecem associadas à conversão, mas exigem cautela por risco de vazamento de informação.

### 3. Teorema de Bayes

O projeto implementa um classificador Bayesiano manual, sem depender de uma biblioteca pronta para essa etapa.

Foram calculadas:

- probabilidades a priori `P(C)`;
- verossimilhanças `P(X|C)`;
- probabilidades a posteriori `P(C|X)`;
- suavização de Laplace para evitar probabilidade zero em categorias pouco frequentes ou não vistas.

As probabilidades do Bayes são salvas em `models/bayes_dados.json` e usadas no dashboard para prever a probabilidade de compra de um novo cliente.

### 4. Algoritmos de classificação

Além do Bayes manual, foram treinados dois modelos supervisionados com `scikit-learn`:

- Regressão Logística;
- Árvore de Decisão.

O pipeline de modelagem inclui:

- separação entre variáveis preditoras e variável alvo;
- one-hot encoding para variáveis categóricas;
- divisão treino/teste com `stratify=y`;
- padronização das variáveis numéricas para a Regressão Logística;
- uso de `class_weight='balanced'` por causa do desbalanceamento da classe alvo;
- salvamento dos modelos treinados em arquivos `.pkl`.

Os modelos salvos são carregados pelo dashboard para comparar as predições com o resultado do Teorema de Bayes.

### 5. Dashboard interativo

O dashboard foi desenvolvido em **Streamlit** e está localizado em `src/app.py`.

Ele possui duas seções principais:

1. **Análise Exploratória de Dados (EDA)**
   - gráficos interativos;
   - filtros por características do cliente e da campanha;
   - interpretações dos principais padrões encontrados.

2. **Simulador e Comparação de Predição**
   - formulário para inserir características de um cliente;
   - predição pelo Teorema de Bayes manual;
   - predição pela Regressão Logística;
   - predição pela Árvore de Decisão;
   - comparação visual das probabilidades de adesão.

## Como executar o projeto

### 1. Clonar o repositório

```bash
git clone <url-do-repositorio>
cd <nome-do-repositorio>
```

### 2. Criar e ativar um ambiente virtual

```bash
python -m venv .venv
source .venv/bin/activate
```

No Windows:

```bash
.venv\Scripts\activate
```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt
```

### 4. Treinar e salvar os modelos

Caso os arquivos da pasta `models/` ainda não existam, execute:

```bash
python src/treinar_salvar_modelos.py
```

### 5. Rodar o dashboard

```bash
streamlit run src/app.py
```

Depois disso, o Streamlit abrirá o dashboard no navegador.

## Tecnologias utilizadas

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- Streamlit
- Plotly
- Jupyter Notebook
- PyArrow

## Arquivos principais

- `notebooks/tratamento_limpeza.ipynb`: limpeza, tratamento e criação de variáveis derivadas.
- `notebooks/eda.ipynb`: análise exploratória e visualizações.
- `notebooks/teorema_bayes.ipynb`: desenvolvimento do cálculo bayesiano.
- `notebooks/pre_processamento_ml.ipynb`: preparação dos dados, treino e avaliação dos modelos.
- `src/teorema_bayes.py`: funções do classificador Bayesiano manual.
- `src/treinar_salvar_modelos.py`: script para treinar e salvar Bayes, Regressão Logística e Árvore de Decisão.
- `src/app.py`: dashboard interativo em Streamlit.

## Referências

- UCI Machine Learning Repository. Bank Marketing Dataset. Disponível em: https://archive.ics.uci.edu/dataset/222/bank+marketing
- Moro, S.; Cortez, P.; Rita, P. **A Data-Driven Approach to Predict the Success of Bank Telemarketing**. Decision Support Systems, 2014.
- Documentação do Scikit-learn: https://scikit-learn.org/
- Documentação do Streamlit: https://streamlit.io/

## Integrantes

- Lucas Ugulino
- Renan Gomes
- Yan Girard

## Declaração de uso de IA generativa

Durante o desenvolvimento do projeto, ferramentas de IA generativa puderam ser utilizadas como apoio para organização das etapas, revisão de conceitos, estruturação da documentação e também para explicação de dúvidas geradas durante o projeto.

