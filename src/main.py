import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from data_processing.csv_transformer import main as transform_csvs
from data_processing.data_integration import integrate_data
from preprocessing.preprocessor import load_data
from analysis.comparative_analysis import analyze_permanence
from modeling.train import train_models

def main():
    """Orquestra o pipeline completo."""
    print("--- INICIANDO PIPELINE DE ANÁLISE DE PERMANÊNCIA ESTUDANTIL ---")

    print("\n[ETAPA 1/4] Transformando CSVs brutos...")
    try:
        transform_csvs()
        print("Transformação de CSVs concluída.")
    except Exception as e:
        print(f"[ERRO] Falha na transformação de CSVs: {e}")
        return

    print("\n[ETAPA 2/4] Carregando e integrando dados...")
    try:
        cursos_df, ies_df, _ = load_data()
        if cursos_df is not None and ies_df is not None:
            final_data = integrate_data(cursos_df, ies_df)
            print("Dados integrados com sucesso.")
        else:
            print("[ERRO] Não foi possível carregar os dados para integração.")
            return
    except Exception as e:
        print(f"[ERRO] Falha na integração de dados: {e}")
        return

    print("\n[ETAPA 3/4] Executando análise comparativa de permanência...")
    try:
        analyze_permanence(final_data.copy())
        print("Análise comparativa concluída.")
    except Exception as e:
        print(f"[ERRO] Falha na análise comparativa: {e}")
        return

    print("\n[ETAPA 4/4] Treinando modelos de previsão de evasão...")
    try:
        final_data.to_csv('data/transformed_data/dados_integrados_finais.csv', sep=';', index=False, encoding='latin1')
        train_models(final_data)
        print("Treinamento de modelos concluído.")
    except Exception as e:
        print(f"[ERRO] Falha no treinamento dos modelos: {e}")
        return
        
    print("\n--- PIPELINE FINALIZADO COM SUCESSO ---")
    print("Relatórios e modelos foram salvos nas pastas 'reports' e 'models'.")

if __name__ == "__main__":
    main()