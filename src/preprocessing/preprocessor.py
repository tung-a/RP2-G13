import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

def preprocess_data(df, target_column='tempo_permanencia'):
    """
    Prepara os dados para treinamento, aplicando encoding e scaling de forma robusta.
    """
    # 1. Remover linhas com dados faltantes para garantir a qualidade
    df.dropna(inplace=True)

    X = df.drop(target_column, axis=1)
    y = df[target_column]

    # 2. Identificar colunas categóricas e numéricas
    categorical_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    numerical_features = X.select_dtypes(include=['int64', 'float64', 'int32', 'float32']).columns.tolist()

    # 3. Identificar e remover colunas categóricas com muitas categorias (alta cardinalidade)
    #    Isso evita a criação de um número excessivo de features.
    high_cardinality_cols = [col for col in categorical_features if X[col].nunique() > 50]
    categorical_features = [col for col in categorical_features if col not in high_cardinality_cols]
    
    print(f"Colunas numéricas identificadas: {numerical_features}")
    print(f"Colunas categóricas identificadas (baixa cardinalidade): {categorical_features}")
    print(f"Colunas categóricas descartadas (alta cardinalidade): {high_cardinality_cols}")

    # 4. Criar os transformadores
    numerical_transformer = StandardScaler()
    categorical_transformer = OneHotEncoder(handle_unknown='ignore')

    # --- CORREÇÃO APLICADA AQUI ---
    # Alterado remainder='passthrough' para remainder='drop'.
    # Isso garante que qualquer coluna não especificada (como as de alta cardinalidade)
    # seja descartada em vez de enviada ao modelo.
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features)
        ],
        remainder='drop'
    )

    # 5. Dividir os dados em treino e teste
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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