import os
from data_processing.data_integration import load_and_integrate_data, split_by_institution_type
from preprocessing.preprocessor import preprocess_data, save_preprocessor
from modeling.train import train_model, evaluate_model, save_model
from analysis.regression_analysis import analyze_feature_importance, save_metrics_report

def main():
    """
    Executa o pipeline completo de machine learning para prever o tempo de permanência.
    """
    # Definição de caminhos
    DATA_PATH = 'data'
    MODELS_PATH = 'models'
    REPORTS_PATH = 'reports/figures'
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(REPORTS_PATH, exist_ok=True)

    # 1. Carga e Integração de Dados (AGORA USANDO DASK POR BAIXO DOS PANOS)
    print("--- Iniciando Etapa 1: Carga e Integração de Dados ---")
    integrated_df = load_and_integrate_data(DATA_PATH)
    
    if integrated_df.empty:
        print("Nenhum dado retornado após a integração. Encerrando o pipeline.")
        return

    df_publica, df_privada = split_by_institution_type(integrated_df)

    # Dicionário para armazenar as métricas de cada modelo
    metrics = {}
    
    datasets = {
        'publica': df_publica,
        'privada': df_privada
    }

    for name, df in datasets.items():
        if df.empty:
            print(f"\n--- DataFrame '{name}' está vazio. Pulando treinamento. ---")
            continue
            
        print(f"\n--- Processando Dados para Instituições do Tipo: {name.upper()} ---")

        # 2. Pré-processamento
        X_train, X_test, y_train, y_test, preprocessor_pipeline = preprocess_data(df)
        save_preprocessor(preprocessor_pipeline, os.path.join(MODELS_PATH, f'preprocessor_{name}.joblib'))

        # 3. Treinamento do Modelo
        model = train_model(X_train, y_train)
        save_model(model, os.path.join(MODELS_PATH, f'permanencia_model_{name}.joblib'))
        
        # 4. Avaliação
        metrics[name] = evaluate_model(model, X_test, y_test)
        
        # 5. Análise
        analyze_feature_importance(model, preprocessor_pipeline, REPORTS_PATH)

    # 6. Salvar Relatório Final de Métricas
    if 'publica' in metrics and 'privada' in metrics:
        save_metrics_report(metrics['publica'], metrics['privada'], 'reports')
    
    print("\n--- Pipeline concluído com sucesso! ---")

if __name__ == '__main__':
    main()