import os
import pandas as pd
from data_processing.data_integration import load_and_integrate_data, split_by_institution_type
from preprocessing.preprocessor import preprocess_data, save_preprocessor
from modeling.train import train_model, evaluate_model, save_model
from analysis.regression_analysis import analyze_feature_importance, plot_combined_feature_importance, save_metrics_report
# Importa a função de análise do tempo ideal
from analysis.comparative_analysis import run_ideal_time_analysis

def main():
    """
    Executa o pipeline completo: integração, análise comparativa e, em seguida,
    o treinamento dos modelos de machine learning.
    """
    # Definição de caminhos
    DATA_PATH = 'data'
    MODELS_PATH = 'models'
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures')
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # 1. Carga e Integração de Dados
    print("--- Iniciando Etapa 1: Carga e Integração de Dados ---")
    integrated_df = load_and_integrate_data(DATA_PATH)
    
    if integrated_df.empty:
        print("Nenhum dado retornado após a integração. Encerrando o pipeline.")
        return

    # 2. Executar a Análise Comparativa com o Tempo Ideal (para gerar relatórios)
    # A função é chamada com uma cópia do DataFrame para garantir que o original não seja modificado
    # com colunas que possam causar fuga de dados no modelo.
    run_ideal_time_analysis(integrated_df.copy(), FIGURES_PATH)

    # 3. Preparar dados para o treinamento dos modelos
    df_publica, df_privada = split_by_institution_type(integrated_df)

    df_publica.to_csv('data/publica_sample.csv', index=False)
    df_privada.to_csv('data/privada_sample.csv', index=False)
    
    metrics = {}
    datasets_graficos = {}
    
    datasets = {
        'publica': df_publica,
        'privada': df_privada
    }

    for name, df in datasets.items():
        if df.empty:
            print(f"\n--- DataFrame '{name}' está vazio. Pulando treinamento. ---")
            continue
            
        print(f"\n--- Processando Dados para Instituições do Tipo: {name.upper()} ---")

        # 4. Pré-processamento (agora com o preprocessor.py corrigido)
        X_train, X_test, y_train, y_test, preprocessor_pipeline = preprocess_data(df)
        save_preprocessor(preprocessor_pipeline, os.path.join(MODELS_PATH, f'preprocessor_{name}.joblib'))

        # 5. Treinamento do Modelo
        model = train_model(X_train, y_train)
        save_model(model, os.path.join(MODELS_PATH, f'permanencia_model_{name}.joblib'))
        
        # 6. Avaliação
        metrics[name] = evaluate_model(model, X_test, y_test)
        
        # 7. Análise de Importância das Features
        datasets_graficos[name] = analyze_feature_importance(model, preprocessor_pipeline, FIGURES_PATH, name)

    plot_combined_feature_importance(datasets_graficos.get('publica'), datasets_graficos.get('privada'), FIGURES_PATH)
    
    # 8. Salvar Relatório Final de Métricas
    if 'publica' in metrics and 'privada' in metrics:
        save_metrics_report(metrics['publica'], metrics['privada'], REPORTS_PATH)
    
    print("\n--- Pipeline concluído com sucesso! ---")

if __name__ == '__main__':
    main()