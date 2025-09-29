# Análise Preditiva da Permanência de Estudantes no Ensino Superior Brasileiro

Este projeto utiliza técnicas de Machine Learning para prever o **tempo de permanência** de estudantes em Instituições de Ensino Superior (IES) no Brasil. Utilizando microdados públicos do Censo da Educação Superior, o pipeline processa e integra os dados, treina múltiplos modelos de regressão e realiza análises aprofundadas para identificar os fatores que mais influenciam a trajetória académica dos alunos, com uma comparação detalhada entre IES públicas e privadas.

## Principais Funcionalidades

- **Pipeline de Dados Automatizado**: O `main.py` orquestra todo o fluxo, desde a limpeza e integração dos dados até ao treino de um modelo base (Random Forest).
- **Otimização de Modelos**: `runtests.py` permite testar e otimizar múltiplos modelos de regressão (`LightGBM`, `GradientBoosting`, `SVR`, `Ridge`) usando `GridSearchCV` e `RandomizedSearchCV`.
- **Análise de Cenários**: `predict.py` gera e analisa milhares de perfis de alunos hipotéticos, guardando as previsões e gerando gráficos sobre o impacto de cada característica.
- **Análise do "Tempo Ideal"**: O projeto calcula a duração padrão de cada curso e compara-a com o tempo de permanência real para classificar os alunos em "Evasão Provável", "Conclusão no Prazo" ou "Atraso".

## Tecnologias Utilizadas

- Python
- Pandas
- Scikit-learn
- Dask (para manipulação de grandes volumes de dados)
- LightGBM
- Matplotlib / Seaborn

## Estrutura do Projeto

```
/
├── data/
│   ├── ces/              # Dados brutos do Censo da Educação Superior
│   ├── IGC/
│   │   └── igc_tratado.csv # Arquivo com os dados do IGC
│   ├── publica_sample.csv  # Amostra de dados gerada pelo main.py
│   └── privada_sample.csv  # Amostra de dados gerada pelo main.py
├── models/               # Modelos treinados (.joblib) são guardados aqui
├── reports/
│   ├── figures/          # Gráficos e visualizações gerados
│   └── *.csv             # Relatórios e resumos de desempenho
├── src/
│   ├── analysis/
│   │   ├── comparative_analysis.py # Compara tempo real vs. ideal
│   │   └── regression_analysis.py  # Analisa importância das features
│   ├── data_processing/
│   │   └── data_integration.py     # Carrega, limpa e integra os dados
│   ├── modeling/
│   │   └── train.py                # Funções para treinar e avaliar modelos
│   ├── preprocessing/
│   │   └── preprocessor.py         # Prepara os dados para o modelo
│   ├── main.py                     # Script principal para executar o pipeline completo
│   ├── predict.py                  # Script para gerar e analisar cenários de previsão
│   └── runtests.py                 # Script para testar e otimizar múltiplos modelos
├── requirements.txt      # Dependências do projeto
└── README.md
```

## Como Executar o Projeto

### 1. Pré-requisitos

- Python 3.8 ou superior
- Git

### 2. Configuração do Ambiente

```bash
# Clone o repositório
git clone https://github.com/seu-usuario/RP2-G13.git
cd RP2-G13

# Crie e ative um ambiente virtual (recomendado)
python -m venv venv
./venv/Scripts/Activate

# Instale as dependências
pip install -r requirements.txt
```

### 3. Estrutura de Dados

Garanta que os seus dados brutos estejam descompactados dentro da pasta `data/`, seguindo a estrutura esperada pelos scripts (ex: `data/ces/...`, `data/IGC/...`).

### 4. Execução do Pipeline Principal

Este é o primeiro passo e o mais importante. Ele processa todos os dados, treina um modelo de base e gera os arquivos de amostra necessários para os testes.

```bash
python src/main.py
```

### 5. Testando e Otimizando Outros Modelos

Após a execução bem-sucedida do `main.py`, use o `runtests.py` para encontrar o melhor modelo para os seus dados.

**Exemplos:**

```bash
# Testar o LightGBM com GridSearchCV
python src/runtests.py --model LightGBM --method grid

# Testar o SVR com RandomizedSearchCV (20 iterações)
python src/runtests.py --model SVR --method random --n_iter 20
```

Os resultados serão guardados num arquivo `.csv` dentro da pasta `reports/`.

### 6. Gerando e Analisando Cenários de Previsão

Este script usa os modelos treinados para simular o impacto de diferentes características.

```bash
python src/predict.py
```

Ele irá gerar o arquivo `reports/prediction_scenarios.csv` e vários gráficos de análise na pasta `reports/figures/`.

## Outputs Gerados

- **`models/`**: Contém os arquivos `.joblib` dos modelos treinados e dos pré-processadores.
- **`reports/`**: Contém os relatórios `.csv` com as métricas de desempenho dos modelos e os cenários de previsão.
- **`reports/figures/`**: Contém todos os gráficos gerados, como a importância das características e as análises comparativas.
