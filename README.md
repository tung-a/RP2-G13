# Análise da Permanência e Evasão no Ensino Superior Brasileiro

Este projeto representa um pipeline completo de análise de dados, projetado para investigar os fatores relacionados à permanência e evasão de estudantes em Instituições de Ensino Superior (IES) no Brasil. Utilizando microdados públicos do Censo da Educação Superior, o projeto processa os dados, realiza análises comparativas e treina modelos de machine learning para prever a probabilidade de evasão.

## Principais Funcionalidades

- **Pipeline de Dados Automatizado:** Um script principal (`main.py`) orquestra todo o fluxo, desde a limpeza dos dados brutos até o treinamento dos modelos.
- **Análise Comparativa:** Gera estatísticas e gráficos que comparam a evasão entre IES públicas e privadas, por área do conhecimento e através de uma métrica de "eficiência de conclusão".
- **Modelagem Preditiva:** Treina e avalia múltiplos modelos de classificação (Regressão Logística, Árvore de Decisão e Random Forest) para prever cursos com alta propensão à evasão.
- **Sistema de Teste e Predição:** Inclui scripts para testar os modelos em lote com diversos cenários e um script interativo para fazer previsões para cursos específicos.

## Estrutura do Projeto

O projeto é modularizado para garantir a clareza e a manutenibilidade do código.

- **`src`**: Pasta principal contendo todo o código-fonte.
  - **`data_processing/`**: Scripts para transformar (`csv_transformer.py`) e integrar (`data_integration.py`) os dados.
  - **`analysis/`**: Contém os diversos scripts de análise:
    - `comparative_analysis.py`: Compara a taxa de evasão entre IES públicas e privadas.
    - `course_area_analysis.py`: Analisa a evasão por grande área do conhecimento.
    - `permanence_efficiency_analysis.py`: Calcula e visualiza a "eficiência de conclusão" por curso.
    - `analyze_test_results.py`: Analisa os resultados gerados pelo script de teste em lote.
  - **`modeling/`**: Script para treinar e salvar os modelos de machine learning (`train.py`).
  - **`main.py`**: Orquestrador que executa o pipeline principal de processamento e treinamento.
  - **`predict.py`**: Script interativo para fazer previsões usando os modelos treinados.
  - **`test_predictor.py`**: Script para testar os modelos em lote com múltiplos cenários.
- **`data/`**: Deve conter os microdados brutos do Censo e do ENEM.
- **`reports/`**: Pasta onde todos os relatórios (gráficos `.png` e análises `.csv`) são salvos.
- **`models/`**: Pasta onde os modelos treinados (`.joblib`) são salvos.
- **`check_columns.py`**: Utilitário para inspecionar os nomes das colunas nos arquivos de dados brutos.

## Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- Git

### 1. Configuração Inicial

Primeiro, clone o repositório e instale as dependências necessárias.

```bash
git clone https://github.com/seu-usuario/RP2-G13.git
cd RP2-G13
pip install -r requirements.txt
```

### 2. Estrutura de Dados

Garanta que os microdados do Censo da Educação Superior estejam descompactados e organizados dentro da pasta `data/`, seguindo a estrutura esperada pelos scripts.

### 3. Executando o Pipeline Principal

O script `main.py` executa o fluxo completo de processamento dos dados e treinamento dos modelos.

```bash
python src/main.py
```

Ao final, as pastas `reports/` e `models/` estarão populadas com os resultados.

### 4. Executando as Análises Adicionais

Após executar o `main.py` pelo menos uma vez, você pode rodar as análises específicas.

```bash
# Para analisar a evasão por área do conhecimento
python src/analysis/course_area_analysis.py

# Para analisar a eficiência de conclusão por curso
python src/analysis/permanence_efficiency_analysis.py
```

### 5. Usando os Modelos Preditivos

Para fazer previsões interativas ou testar os modelos.

```bash
# Para fazer uma previsão para um curso específico
python src/predict.py

# Para rodar o teste em lote e gerar o CSV de comparação
python src/test_predictor.py

# Para analisar os resultados do teste em lote
python src/analysis/analyze_test_results.py
```

## Outputs Gerados

- **Relatórios (`reports/`):** Contém todos os gráficos e tabelas gerados, como a comparação da taxa de evasão, o ranking de cursos por eficiência e os relatórios de classificação dos modelos.
- **Modelos (`models/`):** Contém os arquivos `.joblib` dos modelos treinados, prontos para serem usados para predição.
