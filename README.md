# Análise Preditiva da Permanência Estudantil no Brasil

Este projeto realiza uma análise de dados e treina modelos de machine learning para comparar e prever a evasão de estudantes em instituições de ensino superior (IES) públicas e privadas no Brasil. Utilizando dados públicos do Censo da Educação Superior (INEP), o pipeline processa, analisa e modela os fatores associados à permanência estudantil.

## Objetivos

- **Analisar Comparativamente:** Gerar estatísticas e visualizações para comparar a taxa de evasão entre IES públicas e privadas.
- **Modelagem Preditiva:** Treinar e avaliar diferentes modelos de classificação para prever se um curso terá uma taxa de evasão "alta" ou "baixa".
- **Pipeline Automatizado:** Criar um fluxo de trabalho de dados reproduzível, desde o tratamento dos dados brutos até a geração de relatórios e modelos treinados.

## Estrutura do Projeto

O projeto é organizado em um pipeline de dados com os seguintes módulos:

- `src/data_processing`: Scripts para transformar os dados brutos do Censo (`csv_transformer.py`) e integrar as bases de dados de cursos e IES (`data_integration.py`).
- `src/analysis`: Contém a lógica para a análise estatística e visual comparativa da evasão (`comparative_analysis.py`).
- `src/modeling`: Script responsável pelo treinamento e avaliação dos modelos de machine learning (`train.py`).
- `src/preprocessing`: Módulo para o pré-processamento dos dados.
- `src/main.py`: Orquestrador principal que executa todas as etapas do pipeline em sequência.

## Como Executar

### Pré-requisitos

- Python 3.8 ou superior
- Pip (gerenciador de pacotes do Python)

### 1. Estrutura de Pastas de Dados

Antes de executar, certifique-se de que os dados brutos do Censo da Educação Superior e do ENEM estejam na seguinte estrutura dentro da pasta `data/`:

```
data/
├── ces/
│   ├── microdata_2020/dados/MICRODADOS_CADASTRO_CURSOS_2020.CSV
│   │   └── ... (outros arquivos do Censo 2020)
│   ├── microdata_2021/dados/...
│   ├── microdata_2022/dados/...
│   └── microdata_2023/dados/...
└── enem/
    ├── microdata_2020/MICRODADOS_ENEM_2020.csv
    │   └── ... (outros arquivos do ENEM 2020)
    ├── microdata_2021/...
    ├── microdata_2022/...
    └── microdata_2023/...
```

### 2. Instalação das Dependências

Clone o repositório e instale as bibliotecas necessárias:

```bash
git clone https://github.com/seu-usuario/RP2-G13.git
cd RP2-G13
pip install -r requirements.txt
```

### 3. Execução do Pipeline Completo

Para executar todo o processo, desde a transformação dos dados até o treinamento dos modelos, execute o script `main.py`:

```bash
python src/main.py
```

O script irá executar as seguintes etapas:

- **Transformação dos Dados:** Lê os arquivos CSV brutos, filtra as colunas relevantes e salva os dados tratados em `data/transformed_data/`.
- **Integração dos Dados:** Carrega os dados tratados e os une em um único dataframe.
- **Análise Comparativa:** Gera estatísticas e gráficos comparando a evasão em IES públicas e privadas e os salva na pasta `reports/`.
- **Treinamento dos Modelos:** Treina modelos de Regressão Logística, Árvore de Decisão e Random Forest separadamente para cada tipo de IES. Os modelos treinados são salvos na pasta `models/` e os relatórios de classificação na pasta `reports/`.

## Saídas Geradas (Outputs)

Após a execução bem-sucedida, as seguintes pastas e arquivos serão criados ou atualizados:

- `data/transformed_data/`: Contém os arquivos CSV intermediários após a limpeza e filtragem.
- `reports/`:
  - `estatisticas_evasao.csv`: Tabela com estatísticas descritivas da taxa de evasão.
  - `boxplot_taxa_evasao.png`: Gráfico de boxplot comparando a distribuição da evasão.
  - `histograma_taxa_evasao.png`: Histograma para visualizar a frequência da evasão.
  - `*_classification_report.csv`: Relatórios detalhados do desempenho de cada modelo treinado.
- `models/`:
  - `*.joblib`: Arquivos contendo os objetos dos modelos treinados, prontos para serem carregados e utilizados para predições.
