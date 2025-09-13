import os
import pandas as pd

def find_csv_files(root_dir, pattern):
    """Encontra todos os arquivos CSV que correspondem a um padrão em um diretório."""
    csv_files = []
    for root, _, files in os.walk(root_dir):
        for file in files:
            if file.startswith(pattern) and file.endswith(('.csv', '.CSV')):
                csv_files.append(os.path.join(root, file))
    return csv_files

def process_and_save_csvs(name, root_dir, pattern, columns_to_keep, output_path):
    """
    Lê múltiplos arquivos CSV, filtra as colunas desejadas, concatena os dados
    e salva o resultado em um único arquivo.
    """
    print(f"Iniciando leitura e filtragem dos arquivos de {name}...")
    
    all_dfs = []
    csv_files = find_csv_files(root_dir, pattern)
    
    if not csv_files:
        print(f"!!! AVISO: Nenhum arquivo CSV encontrado para o padrão '{pattern}' em '{root_dir}' !!!")
        return

    for file_path in csv_files:
        print(f"Lendo arquivo: {file_path}")
        try:
            df = pd.read_csv(file_path, sep=';', encoding='latin1', low_memory=False, usecols=columns_to_keep)
            all_dfs.append(df)
        except Exception as e:
            print(f"!!! ERRO ao ler o arquivo {file_path}: {e} !!!")
            print("Verifique se o separador e o encoding estão corretos.")

    if not all_dfs:
        print(f"Nenhum dataframe foi carregado para {name}. O arquivo de saída não será gerado.")
        return

    df_final = pd.concat(all_dfs, ignore_index=True)
    
    # Garante que o diretório de saída exista
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    
    df_final.to_csv(output_path, sep=';', index=False, encoding='latin1')
    print(f"CSVs de {name} transformados e salvos em '{output_path}'\n")

def main():
    """
    Função principal para orquestrar a transformação dos CSVs.
    """
    # --- Processamento dos arquivos de CURSOS ---
    cursos_cols_to_keep = [
        'CO_IES', 'NO_CINE_ROTULO', 'QT_MAT', 'QT_CONC', 'QT_SIT_DESVINCULADO'
    ]
    process_and_save_csvs(
        'CURSOS',
        'data/ces',
        'MICRODADOS_CADASTRO_CURSOS',
        cursos_cols_to_keep,
        'data/transformed_data/cursos_tratados.csv'
    )

    # --- Processamento dos arquivos de IES ---
    ies_cols_to_keep = [
        'CO_IES', 'NO_IES', 'SG_IES',
        'TP_CATEGORIA_ADMINISTRATIVA',  # <<<--- COLUNA ADICIONADA AQUI
        'NO_REGIAO_IES', 'SG_UF_IES', 'NO_MUNICIPIO_IES'
    ]
    process_and_save_csvs(
        'IES',
        'data/ces',
        'MICRODADOS_CADASTRO_IES',
        ies_cols_to_keep,
        'data/transformed_data/ies_tratados.csv'
    )

    # --- Processamento dos arquivos do ENEM ---
    enem_cols_to_keep = [
        'NU_INSCRICAO', 'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL', 'TP_COR_RACA',
        'TP_ESCOLA', 'NU_NOTA_CN', 'NU_NOTA_CH', 'NU_NOTA_LC', 'NU_NOTA_MT',
        'NU_NOTA_REDACAO', 'Q001', 'Q002', 'Q006'
    ]
    process_and_save_csvs(
        'ENEM',
        'data/enem',
        'MICRODADOS_ENEM',
        enem_cols_to_keep,
        'data/transformed_data/enem_tratados.csv'
    )

if __name__ == "__main__":
    main()