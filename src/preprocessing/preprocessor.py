import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(df, target_column='tempo_permanencia', high_cardinality_threshold=50):
    """
    Prepara os dados para treinamento, removendo colunas que causam fuga de dados,
    e aplicando encoding e scaling de forma robusta.
    """

    # Cria uma cópia para evitar SettingWithCopyWarning
    df = df[df['nu_ano_censo'] >= 2019].copy()

    # --- AJUSTES ESPECÍFICOS DE COLUNAS ---
    # 1. Ajustar 'tp_sexo' para booleano: 1 -> True, 2 -> False
    if 'tp_sexo' in df.columns:
        df['tp_sexo'] = df['tp_sexo'].astype('object')
        df.loc[df['tp_sexo'] == 1, 'tp_sexo'] = True
        df.loc[df['tp_sexo'] == 2, 'tp_sexo'] = False
        
        # Converte a coluna para tipo booleano no final
        df['tp_sexo'] = df['tp_sexo'].astype(bool)

    # 2. Garantir que 'tp_cor_raca' seja tratada como categórica (object/category)
    if 'tp_cor_raca' in df.columns:
        df['tp_cor_raca'] = df['tp_cor_raca'].astype('object')

    if 'tp_escola_conclusao_ens_medio' in df.columns:
        df['tp_escola_conclusao_ens_medio'] = df['tp_escola_conclusao_ens_medio'].astype('object')
    
    if 'tp_categoria_administrativa' in df.columns:
        df['tp_categoria_administrativa'] = df['tp_categoria_administrativa'].astype('object')

    if 'tp_modalidade_ensino' in df.columns:
        df['tp_modalidade_ensino'] = df['tp_modalidade_ensino'].astype('object')

    if 'tp_grau_academico' in df.columns:
        df['tp_grau_academico'] = df['tp_grau_academico'].astype('object')

    # --- CORREÇÃO DE DATA LEAKAGE ---
    # Remove colunas que são calculadas a partir do alvo ou que contêm a resposta.
    # Esta é a correção mais importante para obter um modelo válido.
    cols_to_drop = ['nu_ano_censo', 'diferenca_permanencia', 'status_conclusao', 'tipo_ies']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 1. Remover linhas com dados faltantes para garantir a qualidade
    df.dropna(inplace=True)
    print(df.dtypes)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 2. Identificar colunas categóricas e numéricas
    categorical_features = X.select_dtypes(include=['object', 'category','string[pyarrow]']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32','bool']).columns.tolist()

    # 3. Identificar e remover colunas categóricas com muitas categorias (alta cardinalidade)
    high_cardinality_cols = [col for col in categorical_features if X[col].nunique() > high_cardinality_threshold]
    categorical_features = [col for col in categorical_features if col not in high_cardinality_cols]
    
    print(f"Colunas numéricas identificadas: {numerical_features}")
    print(f"Colunas categóricas identificadas (baixa cardinalidade): {categorical_features}")
    print(f"Colunas categóricas descartadas (alta cardinalidade): {high_cardinality_cols}")

    # 4. Criar os transformadores
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # Garante que qualquer coluna não especificada seja descartada
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 5. Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    print(f"\n--- Divisão Treino/Teste ---")
    print(f"Total de amostras restantes: {len(X_train) + len(X_test)}")
    print(f"Amostras para Treinamento (80%): {len(X_train)}")
    print(f"Amostras para Teste (20%): {len(X_test)}")
    
    # 6. Criar e treinar o pipeline de pré-processamento
    pipeline = Pipeline(steps=[('preprocessor', preprocessor)])
    
    X_train_processed = pipeline.fit_transform(X_train)
    X_test_processed = pipeline.transform(X_test)

    print("Pré-processamento concluído.")
    
    return X_train_processed, X_test_processed, y_train, y_test, pipeline

def preprocess_for_kmeans(df_pandas: pd.DataFrame):
    """
    Prepara um DataFrame Pandas para o Scikit-learn MiniBatchKMeans.
    1. Aplica One-Hot Encoding em colunas categóricas.
    2. Aplica StandardScaler em colunas numéricas.
    3. Retorna um Pandas DataFrame/NumPy Array pronto para o modelo e a lista de features.
    """
    # Importar StandardScaler aqui se ainda não estiver no topo
    from sklearn.preprocessing import StandardScaler
    
    print(f"--- Iniciando pré-processamento para K-Means em {len(df_pandas)} registros... ---")
    
    # 1. Identificar colunas (MANTENHA ESTA LÓGICA)
    # Colunas que são numéricas e devem ser escaladas
    numerical_cols = [
        'in_apoio_social', 'in_financiamento_estudantil', 'nu_carga_horaria',
        'pib', 'inscritos_por_vaga', 'duracao_ideal_anos', 'igc', 'taxa_integralizacao'
    ]
    
    # Colunas que são categóricas por natureza
    categorical_cols = [
        'nu_ano_censo', 'tp_cor_raca', 'tp_sexo', 'faixa_etaria',
        'tp_escola_conclusao_ens_medio', 'tp_modalidade_ensino',
        'tp_grau_academico', 'nm_categoria', 'sigla_uf_curso', 
        'no_regiao_ies'
    ]
    
    existing_numerical = [col for col in numerical_cols if col in df_pandas.columns]
    existing_categorical = [col for col in categorical_cols if col in df_pandas.columns]

    print(f"Colunas numéricas para escalar: {existing_numerical}")
    print(f"Colunas categóricas para One-Hot-Encoding: {existing_categorical}")

    df_to_process = df_pandas.copy()

    # 2. Aplicar Scaling nas numéricas
    scaler = StandardScaler()
    scaled_values = scaler.fit_transform(df_to_process[existing_numerical])
    
    df_scaled = pd.DataFrame(
        scaled_values, 
        columns=existing_numerical, 
        index=df_to_process.index
    )
    
    # 3. Aplicar One-Hot Encoding
    for col in existing_categorical:
        df_to_process[col] = df_to_process[col].astype('category')
            
    df_dummies = pd.get_dummies(
        df_to_process[existing_categorical], 
        dummy_na=False, 
        drop_first=False
    )
    
    # 4. Combinar os DataFrames processados
    df_final_processed = pd.concat([df_scaled, df_dummies], axis=1)
    
    # Converter para um tipo numérico compatível com Scikit-learn (float32 é suficiente)
    df_final_processed = df_final_processed.astype('float32')

    # 5. Retornar os valores NumPy (Array) diretamente, sem Dask.
    # O MiniBatchKMeans aceita DataFrames, mas o array NumPy é o formato mais universal
    numpy_array = df_final_processed.values 
    
    print(f"Processamento concluído. Forma do array NumPy: {numpy_array.shape}")
    
    # Retornar o array NumPy e os nomes das colunas
    return numpy_array, df_final_processed.columns.tolist()

def save_preprocessor(pipeline, path):
    """Salva o pipeline de pré-processamento."""
    print(f"Salvando o pipeline de pré-processamento em {path}")
    joblib.dump(pipeline, path)