import pandas as pd
import os

def get_csv_headers(file_path):
    """
    Lê a primeira linha de um arquivo CSV de forma eficiente para obter os nomes das colunas.
    """
    try:
        # Lê apenas o cabeçalho (nrows=0) para ser rápido e economizar memória
        header = pd.read_csv(file_path, sep=';', encoding='latin1', nrows=0).columns.tolist()
        return header
    except Exception as e:
        return f"Não foi possível ler o arquivo. Erro: {e}"

def main():
    """
    Script principal para inspecionar os cabeçalhos dos arquivos de dados brutos.
    """
    print("--- INSPECIONANDO NOMES DAS COLUNAS NOS ARQUIVOS DE DADOS BRUTOS ---")

    # Defina aqui os caminhos para os arquivos que você quer inspecionar
    # Vamos verificar um arquivo de cada tipo (Cursos, IES, ENEM) para um ano específico.
    files_to_check = {
        "Cursos 2020": os.path.join('data', 'ces', 'microdata_2020', 'dados', 'MICRODADOS_CADASTRO_CURSOS_2020.CSV'),
        "IES 2020"   : os.path.join('data', 'ces', 'microdata_2020', 'dados', 'MICRODADOS_CADASTRO_IES_2020.CSV'),
        "ENEM 2020"  : os.path.join('data', 'enem', 'microdata_2020', 'MICRODADOS_ENEM_2020.csv')
    }

    for name, path in files_to_check.items():
        print(f"\n--- Colunas encontradas em: {name} ({path}) ---")
        
        if not os.path.exists(path):
            print("ARQUIVO NÃO ENCONTRADO.")
            continue
            
        columns = get_csv_headers(path)
        
        if isinstance(columns, list):
            for col in columns:
                print(f"- {col}")
        else:
            print(columns) # Imprime a mensagem de erro, se houver

    print("\n--- INSPEÇÃO FINALIZADA ---")
    print("Use esta lista para verificar os nomes exatos das colunas necessárias para o seu pipeline.")

if __name__ == "__main__":
    main()