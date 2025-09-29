from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import pandas as pd

def train_model(X_train, y_train,params= None):
    """
    Treina um modelo de regressão.
    
    Args:
        X_train: Features de treinamento.
        y_train: Variável alvo de treinamento.

    Returns:
        O modelo treinado.
    """
    if params is None:
        params = {}
    
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