import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

def train_models(data):
    """
    Treina e avalia múltiplos modelos de classificação para prever a evasão,
    utilizando features do Censo da Educação Superior.
    """
    print("--- Iniciando Treinamento dos Modelos (Foco no Censo) ---")
    os.makedirs('models', exist_ok=True)

    tipos_ies = {'publica': [1, 2, 3], 'privada': [4, 5]}

    for tipo, ids in tipos_ies.items():
        print(f"\n--- Treinando modelos para IES do tipo: {tipo.upper()} ---")
        
        subset_df = data[data['TP_CATEGORIA_ADMINISTRATIVA'].isin(ids)].copy().dropna()
        
        if subset_df.empty or len(subset_df) < 10:
            print(f"Dados insuficientes para IES do tipo {tipo}. Pulando.")
            continue

        y = subset_df['ALTA_EVASAO']
        X = subset_df.drop(columns=['ALTA_EVASAO', 'CO_IES', 'NO_MUNICIPIO'])
        
        # Identificar colunas categóricas e numéricas para pré-processamento
        categorical_features = X.select_dtypes(include=['object']).columns
        numeric_features = X.select_dtypes(include=['number']).columns

        # Criar um pré-processador para converter texto em números (OneHotEncoder)
        preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ], remainder='passthrough')

        # Definindo os modelos
        models = {
            'RegressaoLogistica': LogisticRegression(max_iter=1000, random_state=42),
            'ArvoreDecisao': DecisionTreeClassifier(random_state=42, max_depth=10),
            'RandomForest': RandomForestClassifier(random_state=42, n_estimators=100, max_depth=10)
        }

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
        
        for name, model in models.items():
            # Criar o pipeline completo: pré-processamento + modelo
            pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('classifier', model)])
            
            print(f"Treinando {name}...")
            pipeline.fit(X_train, y_train)
            
            y_pred = pipeline.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            print(f"Acurácia do {name} ({tipo}): {acc:.4f}")
            print(classification_report(y_test, y_pred))
            
            model_path = f"models/{name}_{tipo}.joblib"
            joblib.dump(pipeline, model_path)
            print(f"Modelo salvo em: {model_path}")

    print("\n--- Fim do Treinamento ---")