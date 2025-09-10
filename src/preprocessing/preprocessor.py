import pandas as pd
from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer

# Caminhos para os arquivos de dados JÁ TRATADOS pelo csv_transformer.py
csv_cursos_path = 'data/transformed_data/cursos_tratados.csv'
csv_ies_path = 'data/transformed_data/ies_tratados.csv'
csv_enem_path = 'data/transformed_data/enem_tratados.csv'

def load_data():
    """Carrega os arquivos CSV tratados."""
    try:
        cursos_df = pd.read_csv(csv_cursos_path, sep=';', encoding='latin1', low_memory=False)
        ies_df = pd.read_csv(csv_ies_path, sep=';', encoding='latin1', low_memory=False)
        enem_df = pd.read_csv(csv_enem_path, sep=';', encoding='latin1', low_memory=False)
        return cursos_df, ies_df, enem_df
    except FileNotFoundError as e:
        print(f"!!! ERRO: Arquivo não encontrado: {e.filename} !!!")
        print("Por favor, execute o script 'src/csv_transformer.py' primeiro para gerar os arquivos tratados.\n")
        return None, None, None

def preprocess_data(cursos_df, ies_df, enem_df):
    """
    Executa o pré-processamento e a normalização dos dados,
    seguindo as diretrizes do arquivo article.tex.
    """
    print("--- Iniciando Pré-processamento e Transformação ---")

    # Definindo colunas numéricas e categóricas
    numeric_features_enem = ['NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO']
    categorical_features_enem = [
        'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
        'TP_ESCOLA', 'Q001', 'Q002', 'Q006'
    ]

    # Pipeline para variáveis numéricas
    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', MinMaxScaler())
    ])

    # Pipeline para variáveis categóricas
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='most_frequent')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
    ])

    # Criando o pré-processador com o ColumnTransformer
    preprocessor_enem = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features_enem),
            ('cat', categorical_transformer, categorical_features_enem)
        ],
        remainder='passthrough'
    )

    print("\nPré-processando dados do ENEM...")
    enem_processed = preprocessor_enem.fit_transform(enem_df)
    print("Dados do ENEM processados com sucesso.")

    # --- INÍCIO DA SEÇÃO CORRIGIDA ---

    # Obter os novos nomes das colunas após o One-Hot Encoding
    ohe_feature_names = preprocessor_enem.named_transformers_['cat']['onehot'].get_feature_names_out(categorical_features_enem).tolist()

    # Identificar as colunas do remainder na ordem correta
    # O ColumnTransformer coloca as colunas do remainder no final
    processed_cols = numeric_features_enem + categorical_features_enem
    remainder_cols = [col for col in enem_df.columns if col not in processed_cols]

    # Construir a lista final de nomes de colunas na ordem correta
    all_feature_names = numeric_features_enem + ohe_feature_names + remainder_cols

    # --- FIM DA SEÇÃO CORRIGIDA ---

    # Criar o DataFrame final
    enem_processed_df = pd.DataFrame(enem_processed, columns=all_feature_names)

    print("\nPré-visualização do DataFrame do ENEM processado e normalizado:")
    print(enem_processed_df.head())
    print(f"\nFormato do DataFrame processado: {enem_processed_df.shape}")
    print("\n--- Fim do Pré-processamento ---")

    return enem_processed_df


if __name__ == "__main__":
    cursos_df, ies_df, enem_df = load_data()
    if enem_df is not None:
        processed_enem_data = preprocess_data(cursos_df, ies_df, enem_df)