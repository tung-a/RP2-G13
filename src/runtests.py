import os
import argparse
import pandas as pd
from datetime import datetime
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR
from sklearn.linear_model import Ridge
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform
from modeling.train import train_model, evaluate_model, save_model
from preprocessing.preprocessor import preprocess_data, save_preprocessor
from lightgbm import LGBMRegressor

# Carrega os dados de amostra gerados pelo main.py
datasets = {
    'publica': pd.read_csv('data/publica_sample.csv'),
    'privada': pd.read_csv('data/privada_sample.csv')
}

# Argumentos de linha de comando
parser = argparse.ArgumentParser(description='Run Grid Search or Random Search with different models.')
parser.add_argument('--model', type=str, default='RandomForest',
                    choices=['RandomForest', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge'],
                    help='Choose the model for the search.')
parser.add_argument('--method', type=str, default='grid',
                    choices=['grid', 'random'],
                    help='Choose the search method: "grid" for GridSearchCV or "random" for RandomizedSearchCV.')
parser.add_argument('--n_iter', type=int, default=20,
                    help='Number of iterations for RandomizedSearchCV.')

args = parser.parse_args()

# Define o modelo e os hiperparâmetros com base na escolha do usuário
if args.model == 'RandomForest':
    model = RandomForestRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [400, 500 , 600],
        'max_depth': [None],
        'min_samples_split': [2,3],
        'min_samples_leaf': [1, 2],
        'max_features': ['sqrt', None],
        'criterion': ['squared_error']
    }
    param_distributions = {
        'n_estimators': randint(100, 500),
        'max_depth': [10, 20, None],
        'min_samples_split': randint(2, 11),
        'min_samples_leaf': randint(1, 5),
    }
elif args.model == 'LightGBM':
    model = LGBMRegressor(random_state=42, n_jobs=-1)
    param_grid = {
        'n_estimators': [200, 300, 500],
        'learning_rate': [0.05, 0.1],
        'num_leaves': [31, 70],
        'max_depth': [20, -1],
        'min_child_samples': [20, 50],
        'reg_alpha': [0.0, 0.1],
        'reg_lambda': [0.0, 0.1]
    }
    param_distributions = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'num_leaves': randint(20, 150),
        'max_depth': [10, -1],
    }
elif args.model == 'GradientBoosting':
    model = GradientBoostingRegressor(random_state=42)
    param_grid = {
        'n_estimators': [100, 200, 300],
        'learning_rate': [0.01, 0.05, 0.1],
        'max_depth': [3, 5, 8],
        'subsample': [0.7, 0.9, 1.0],
        'min_samples_split': [5, 10, 20]
    }
    param_distributions = {
        'n_estimators': randint(100, 500),
        'learning_rate': uniform(0.01, 0.3),
        'max_depth': randint(3, 10),
    }
elif args.model == 'SVR':
    model = SVR()
    param_grid = {
        'kernel': ['rbf', 'linear'],
        'C': [0.1, 1, 10],
        'epsilon': [0.1, 0.2],
    }
    param_distributions = {
        'kernel': ['rbf', 'linear'],
        'C': uniform(0.1, 10),
        'epsilon': uniform(0.05, 0.5),
    }
elif args.model == 'Ridge':
    model = Ridge(random_state=42)
    param_grid = {
        'alpha': [0.1, 1.0, 10.0],
        'solver': ['auto', 'svd', 'lsqr'],
    }
    param_distributions = {
        'alpha': uniform(0.1, 100),
        'solver': ['auto', 'svd', 'lsqr'],
    }

# Lista para armazenar os resultados
all_results = []

for name, df in datasets.items():
    if df.empty:
        print(f"\n--- DataFrame '{name}' está vazio. Pulando treinamento. ---")
        continue

    X_train, X_test, y_train, y_test, preprocessor_pipeline = preprocess_data(df, target_column='taxa_integralizacao')
    
    if args.method == 'grid':
        print(f"Iniciando GridSearchCV para o modelo {args.model}...")
        search = GridSearchCV(estimator=model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, verbose=1)
    else: # random
        print(f"Iniciando RandomizedSearchCV para o modelo {args.model} com {args.n_iter} iterações...")
        search = RandomizedSearchCV(estimator=model, param_distributions=param_distributions, n_iter=args.n_iter, cv=3, scoring='neg_mean_squared_error', n_jobs=-1, random_state=42, verbose=1)

    print(f"\n--- Processando Dados para Instituições do Tipo: {name.upper()} ---")
    search.fit(X_train, y_train)
    
    best_model = search.best_estimator_
    print(f"Melhores hiperparâmetros: {search.best_params_}")

    metrics = evaluate_model(best_model, X_test, y_test)
    
    result = {
        'model': args.model,
        'dataset': name,
        'search_method': args.method,
        'n_iterations': args.n_iter if args.method == 'random' else 'N/A',
        'best_params': str(search.best_params_),
        'best_score': search.best_score_,
        'r2_score': metrics['r2'],
        'mse': metrics['mse']
    }
    all_results.append(result)

    # Salva o arquivo .joblib na pasta 'models'
    MODELS_PATH = 'models'
    os.makedirs(MODELS_PATH, exist_ok=True)
    model_path = os.path.join(MODELS_PATH, f'{args.model}_{name}_best.joblib')
    save_model(best_model, model_path)
    print(f"Melhor modelo salvo em: {model_path}")

# Salva o resumo .csv na pasta 'reports'
REPORTS_PATH = 'reports'
os.makedirs(REPORTS_PATH, exist_ok=True)
timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
results_df = pd.DataFrame(all_results)
results_csv_path = os.path.join(REPORTS_PATH, f'model_performance_summary_{timestamp}.csv')
results_df.to_csv(results_csv_path, index=False)
print(f"\nResumo de desempenho de todos os modelos salvo em: {results_csv_path}")