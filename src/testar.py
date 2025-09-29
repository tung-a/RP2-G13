import os
import pandas as pd
from data_processing.data_integration import load_and_integrate_data, split_by_institution_type, load_and_integrate_data2
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

    # 1. Carga e Integração de Dados
    print("--- Iniciando Etapa 1: Carga e Integração de Dados ---")
    integrated_df = load_and_integrate_data2(DATA_PATH)
    print(integrated_df)

    if integrated_df.empty:
        print("Nenhum dado retornado após a integração. Encerrando o pipeline.")
        return
    
if __name__ == '__main__':
    main()
    