import os
import sys

# Adiciona o diretório 'src' ao path do Python para permitir importações diretas
# Isso garante que o script funcione independentemente de onde for executado
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Importa as funções principais dos outros scripts
from data_processing.csv_transformer import main as transform_csvs
from preprocessing.preprocessor import load_data, preprocess_data
from previewer import preview_dataframe

def main():
    """
    Orquestra todo o pipeline de processamento de dados, desde a transformação inicial
    até o pré-processamento final para modelagem.
    """
    print("--- INICIANDO PIPELINE DE PROCESSAMENTO DE DADOS ---")

    # --- Etapa 1: Transformar os dados brutos ---
    # Executa o script para ler os CSVs originais, filtrar colunas e
    # salvar os arquivos tratados em 'data/transformed_data/'.
    print("\n[ETAPA 1/3] Executando a transformação de CSVs brutos...")
    try:
        transform_csvs()
        print("[ETAPA 1/3] Transformação de CSVs concluída com sucesso.")
    except Exception as e:
        print(f"[ERRO] A etapa de transformação de CSVs falhou: {e}")
        return # Interrompe a execução se a primeira etapa falhar

    # --- Etapa 2: Visualizar os dados tratados ---
    # Executa uma rápida visualização dos dados que acabaram de ser criados.
    print("\n[ETAPA 2/3] Gerando preview dos dados tratados...")
    try:
        # Caminhos dos arquivos gerados na Etapa 1
        csv_cursos_path = 'data/transformed_data/cursos_tratados.csv'
        csv_ies_path = 'data/transformed_data/ies_tratados.csv'
        csv_enem_path = 'data/transformed_data/enem_tratados.csv'
        
        preview_dataframe("Cursos Tratados", csv_cursos_path)
        preview_dataframe("IES Tratados", csv_ies_path)
        preview_dataframe("ENEM Tratados", csv_enem_path)
        print("[ETAPA 2/3] Preview gerado com sucesso.")
    except Exception as e:
        print(f"[ERRO] A etapa de preview falhou: {e}")
        return

    # --- Etapa 3: Pré-processar os dados para modelagem ---
    # Carrega os dados tratados e aplica as transformações finais como
    # normalização e one-hot encoding.
    print("\n[ETAPA 3/3] Executando o pré-processamento para modelagem...")
    try:
        cursos_df, ies_df, enem_df = load_data()
        if enem_df is not None:
            # Foco no ENEM, conforme definido no script preprocessor.py
            processed_enem_data = preprocess_data(cursos_df, ies_df, enem_df)
            print("Dados do ENEM prontos para a etapa de modelagem.")
            # Aqui você poderia salvar o 'processed_enem_data' em um novo arquivo, se quisesse
            # processed_enem_data.to_csv('data/final_data/enem_final.csv', index=False)
            
            print("[ETAPA 3/3] Pré-processamento concluído com sucesso.")
        else:
            print("[ERRO] Não foi possível carregar os dados para o pré-processamento.")
            return
            
    except Exception as e:
        print(f"[ERRO] A etapa de pré-processamento falhou: {e}")
        return
        
    print("\n--- PIPELINE FINALIZADO COM SUCESSO ---")
    print("Próximos passos: integração dos dados e treinamento dos modelos.")


if __name__ == "__main__":
    main()