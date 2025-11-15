from sklearn.ensemble import RandomForestRegressor
from kmodes.kprototypes import KPrototypes
from sklearn.metrics import mean_squared_error, r2_score
import pandas as pd
import numpy as np
import joblib
import logging

logger = logging.getLogger(__name__)

from sklearn.cluster import MiniBatchKMeans
import dask.array as da

def train_model(X_train, y_train, params= None, institution_type:str = ''):
    """
    Treina um modelo de regressão.
    """
    print(institution_type)
    if params is None:
        if institution_type == 'privada':
            params = {
                'bootstrap': True, 
                'criterion': 'squared_error',
                'max_depth': 100, 
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_features': None,
                'n_estimators': 600
            }
        else:
            params = {
                'bootstrap': True, 
                'criterion': 'squared_error',
                'max_depth': None, 
                'min_samples_leaf': 1,
                'min_samples_split': 2,
                'max_features': None,
                'n_estimators': 700
            }
    
    if 'n_estimators' not in params:
        params['n_estimators'] = 100

    model = RandomForestRegressor(**params, random_state=42, n_jobs=-1)
    print("Iniciando o treinamento do modelo RandomForestRegressor...")
    model.fit(X_train, y_train)
    print("Treinamento finalizado.")
    return model

def evaluate_model(model, X_test, y_test):
    """
    Avalia o modelo de regressão e retorna as métricas.
    """
    predictions = model.predict(X_test)
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)
    
    print(f"Mean Squared Error (MSE): {mse:.2f}")
    print(f"R-squared (R2): {r2:.2f}")
    
    return {'mse': mse, 'r2': r2}

def save_model(model, path):
    """Salva o modelo treinado."""
    print(f"Salvando modelo em {path}")
    joblib.dump(model, path)

def train_kmeans_model(X_data, n_clusters: int, random_state: int = 42):
    """
    Treina um modelo MiniBatchKMeans do Scikit-learn.

    Args:
        X_data (np.ndarray ou pd.DataFrame): Os dados pré-processados (numéricos).
        n_clusters (int): O número de clusters (k).
        random_state (int): Seed para reprodutibilidade.

    Returns:
        sklearn.cluster.MiniBatchKMeans: O modelo K-Means treinado.
    """
    logger.info(f"Iniciando o treinamento do MiniBatchKMeans (k={n_clusters})...")
    
    # Usa MiniBatchKMeans para melhor escalabilidade que KMeans
    # O MiniBatchKMeans usa uma abordagem mais eficiente para grandes datasets
    kmeans_model = MiniBatchKMeans(
        n_clusters=n_clusters, 
        random_state=random_state,
        batch_size=256,
        n_init='auto'
    )
    
    # Treina o modelo
    kmeans_model.fit(X_data)
    
    logger.info("Treinamento do MiniBatchKMeans concluído.")
    return kmeans_model

def train_kprototypes(data_matrix, cat_indices, n_clusters, random_state: int = 42):
    """
    Roda o algoritmo K-Prototypes nos dados preparados.
    """
    print(f"\nIniciando K-Prototypes com {n_clusters} clusters...")
    
    kproto = KPrototypes(n_clusters=n_clusters, 
                         init='Cao', 
                         n_init=10, 
                         verbose=0,
                         n_jobs=-1,
                         random_state= random_state)
    
    clusters = kproto.fit_predict(data_matrix, categorical=cat_indices)
    
    print("Clusterização concluída.")
    
    return kproto, clusters

def predict_clusters(model: MiniBatchKMeans, X_data) -> np.ndarray:
    """
    Prevê os clusters para novos dados usando o modelo MiniBatchKMeans treinado.
    
    Args:
        model (MiniBatchKMeans): O modelo K-Means treinado.
        X_data (np.ndarray ou pd.DataFrame): Os dados de entrada.

    Returns:
        np.ndarray: Um array NumPy com os rótulos (labels) dos clusters.
    """
    logger.info("Prevendo clusters (MiniBatchKMeans)...")
    
    # O .predict() do scikit-learn retorna um array NumPy (e não um Dask Array)
    labels_numpy = model.predict(X_data)
    
    logger.info(f"Previsão de clusters concluída. Total de labels: {len(labels_numpy)}")
    return labels_numpy