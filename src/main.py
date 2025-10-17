import os
import logging
import argparse
import pandas as pd
from data_processing.data_integration import load_and_integrate_data, split_by_institution_type
from preprocessing.preprocessor import preprocess_data, save_preprocessor
from modeling.train import train_model, evaluate_model, save_model
from analysis.regression_analysis import analyze_feature_importance, plot_combined_feature_importance, save_metrics_report
# Importa a função de análise do tempo ideal
from analysis.comparative_analysis import run_ideal_time_analysis
from utils.log_config import setup_logging, get_dated_log_filename

LOG_LEVEL = logging.INFO


def parse_args():
     
    parser = argparse.ArgumentParser(
        description="Executa o pipeline completo de processamento de dados e ML."
    )
    
    parser.add_argument(
        '--skip-integration',
        '-s',
        action='store_true',  # Armazena True se a flag estiver presente
        help="Se ativado, pula a etapa de carga/integração (load_and_integrate_data) "
             "e carrega os DataFrames de amostra (publica_sample.csv e privada_sample.csv) diretamente.",
        default=False
    )

    parser.add_argument(
        '--disable-logging',
        '-l',
        action='store_true',
        help="Desativa a criação de logs (seta o nível para CRITICAL).",
        default=False
    )

    return parser.parse_args()
    
def main():
    """
    Executa o pipeline completo: integração, análise comparativa e, em seguida,
    o treinamento dos modelos de machine learning.
    """

    args = parse_args()

    # Definição de caminhos
    DATA_PATH = 'data'
    MODELS_PATH = 'models'
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures')
    
    os.makedirs(MODELS_PATH, exist_ok=True)
    os.makedirs(FIGURES_PATH, exist_ok=True)

    # Log caso ache necessário
    # Obs integrate database não mostra logs detalhados por padrao
    log_name = get_dated_log_filename(prefix="run")
    
    # O logger raiz é retornado
    root_logger = setup_logging(
        log_filename=log_name, 
        log_level=LOG_LEVEL,
        stream_level=LOG_LEVEL,
        enable_logging=not args.disable_logging # Controlado pelo argumento
    )
    
    logger = logging.getLogger('main')
    logger.info("Iniciando o script principal...")

    if args.skip_integration:
        logger.warning("Parâmetro --skip-integration ativado. Carregando DataFrames de arquivos de amostra.")
        print("--- Iniciando Etapa 1: Carregando DataFrames de Amostra (CSV) ---")
        
        # Caminhos dos arquivos de amostra salvos na execução anterior
        publica_sample_path = os.path.join(DATA_PATH, 'publica_sample.csv')
        privada_sample_path = os.path.join(DATA_PATH, 'privada_sample.csv')
        
        try:
            df_publica = pd.read_csv(publica_sample_path)
            df_privada = pd.read_csv(privada_sample_path)
            logger.info(f"Dados públicos carregados: {len(df_publica)} linhas.")
            logger.info(f"Dados privados carregados: {len(df_privada)} linhas.")

            # Recria o integrated_df apenas para rodar a análise comparativa (concatena os dois)
            integrated_df = pd.concat([df_publica, df_privada], ignore_index=True)

        except FileNotFoundError as e:
            logger.error(f"Erro ao carregar arquivos de amostra: {e}. Certifique-se de que eles foram salvos em uma execução anterior. Encerrando.", exc_info=True)
            return
        
    else:
        # 1. Carga e Integração de Dados (Lógica original)
        logger.info("Iniciando Etapa 1: Carga e Integração de Dados (Completa).")
        print("--- Iniciando Etapa 1: Carga e Integração de Dados ---")
        integrated_df = load_and_integrate_data(DATA_PATH)
        
        if integrated_df.empty:
            logger.error("Nenhum dado retornado após a integração. Encerrando o pipeline.")
            return

        # 3. Preparar dados para o treinamento dos modelos
        df_publica, df_privada = split_by_institution_type(integrated_df)

        # Salva os DataFrames processados para uso futuro com --skip-integration
        df_publica.to_csv(os.path.join(DATA_PATH, 'publica_sample.csv'), index=False)
        df_privada.to_csv(os.path.join(DATA_PATH, 'privada_sample.csv'), index=False)
        logger.info("DataFrames públicos e privados salvos como amostra para reuso.")
    
    # --- Continuação do Pipeline (independente da origem dos dados) ---
    
    # 2. Executar a Análise Comparativa com o Tempo Ideal
    run_ideal_time_analysis(integrated_df.copy(), FIGURES_PATH)
    logger.info("Análise Comparativa concluída.")

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

        # 4. Pré-processamento 
        X_train, X_test, y_train, y_test, preprocessor_pipeline = preprocess_data(df,target_column='taxa_integralizacao')
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