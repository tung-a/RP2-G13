import pandas as pd

# --- Colunas a serem mantidas ---

# Colunas do Censo da Educação Superior (Cursos)
cursos_cols_to_keep = [
    'CO_IES', 'NO_REGIAO', 'SG_UF', 'NO_MUNICIPIO', 'TP_CATEGORIA_ADMINISTRATIVA',
    'TP_REDE', 'TP_MODALIDADE_ENSINO', 'NO_CINE_AREA_GERAL', 'IN_GRATUITO',
    'QT_SIT_DESVINCULADO', 'QT_MAT', 'QT_CONC'
]

# Colunas do Censo da Educação Superior (IES)
ies_cols_to_keep = [
    'CO_IES', 'NO_REGIAO_IES', 'SG_UF_IES', 'NO_MUNICIPIO_IES',
    'TP_ORGANIZACAO_ACADEMICA', 'TP_CATEGORIA_ADMINISTRATIVA'
]

# Colunas do ENEM
enem_cols_to_keep = [
    'NU_INSCRICAO', 'NU_ANO', 'TP_FAIXA_ETARIA', 'TP_SEXO', 'TP_ESTADO_CIVIL',
    'TP_COR_RACA', 'TP_NACIONALIDADE', 'TP_ST_CONCLUSAO', 'TP_ANO_CONCLUIU',
    'TP_ESCOLA', 'TP_ENSINO', 'IN_TREINEIRO', 'NU_NOTA_CN', 'NU_NOTA_CH',
    'NU_NOTA_LC', 'NU_NOTA_MT', 'NU_NOTA_REDACAO', 'Q001', 'Q002', 'Q003',
    'Q004', 'Q005', 'Q006', 'Q007', 'Q008', 'Q009', 'Q010', 'Q011', 'Q012',
    'Q013', 'Q014', 'Q015', 'Q016', 'Q017', 'Q018', 'Q019', 'Q020', 'Q021',
    'Q022', 'Q023', 'Q024', 'Q025'
]


# --- Caminhos dos Arquivos ---

csv_cursos_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_CURSOS_20' + str(i) + '.csv' for i in range(20, 24)]
csv_IES_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_IES_20' + str(i) + '.csv' for i in range(20, 24)]
csv_enem_paths = ['data/enem/microdata_20' + str(i) + '/MICRODADOS_ENEM_20' + str(i) + '.csv' for i in range(20, 24)]

def read_and_filter_csvs(paths, columns_to_keep):
    """
    Lê uma lista de arquivos CSV, mantendo apenas as colunas especificadas.
    """
    dfs = []
    for path in paths:
        try:
            print(f"Lendo arquivo: {path}")
            # Use o parâmetro 'usecols' para carregar apenas as colunas de interesse
            df = pd.read_csv(
                path,
                sep=';',
                encoding='latin1',
                low_memory=False,
                usecols=columns_to_keep
            )
            dfs.append(df)
        except FileNotFoundError:
            print(f"Arquivo não encontrado: {path}")
        except ValueError as e:
            print(f"Erro de valor ao ler {path}: {e}. Algumas colunas podem não existir neste arquivo.")

    if not dfs:
        return pd.DataFrame() # Retorna um DataFrame vazio se nenhum arquivo foi lido
        
    return pd.concat(dfs, ignore_index=True)

def main():
    print("Iniciando leitura e filtragem dos arquivos de CURSOS...")
    cursos_df = read_and_filter_csvs(csv_cursos_paths, cursos_cols_to_keep)
    cursos_df.to_csv('data/transformed_data/cursos_tratados.csv', index=False, sep=';')
    print("CSVs de cursos transformados e salvos em 'data/transformed_data/cursos_tratados.csv'")

    print("\nIniciando leitura e filtragem dos arquivos de IES...")
    ies_df = read_and_filter_csvs(csv_IES_paths, ies_cols_to_keep)
    ies_df.to_csv('data/transformed_data/ies_tratados.csv', index=False, sep=';')
    print("CSVs de IES transformados e salvos em 'data/transformed_data/ies_tratados.csv'")

    print("\nIniciando leitura e filtragem dos arquivos do ENEM...")
    enem_df = read_and_filter_csvs(csv_enem_paths, enem_cols_to_keep)
    enem_df.to_csv('data/transformed_data/enem_tratados.csv', index=False, sep=';')
    print("CSVs do ENEM transformados e salvos em 'data/transformed_data/enem_tratados.csv'")


if __name__ == "__main__":
    main()
    print("\nProcessamento concluído.")