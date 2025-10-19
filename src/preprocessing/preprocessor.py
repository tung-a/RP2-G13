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

    print(df.dtypes)
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

    # --- CORREÇÃO DE DATA LEAKAGE ---
    # Remove colunas que são calculadas a partir do alvo ou que contêm a resposta.
    # Esta é a correção mais importante para obter um modelo válido.
    cols_to_drop = ['nu_ano_censo', 'diferenca_permanencia', 'status_conclusao', 'tipo_ies']
    df = df.drop(columns=[col for col in cols_to_drop if col in df.columns], errors='ignore')

    # 1. Remover linhas com dados faltantes para garantir a qualidade
    df.dropna(inplace=True)

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

def save_preprocessor(pipeline, path):
    """Salva o pipeline de pré-processamento."""
    print(f"Salvando o pipeline de pré-processamento em {path}")
    joblib.dump(pipeline, path)