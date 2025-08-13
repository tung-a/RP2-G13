import pandas as pd

csv_cursos_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_CURSOS_20' + str(i) + '.csv' for i in range(19, 24)]
csv_IES_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_IES_20' + str(i) + '.csv' for i in range(19, 24)]
csv_enem_paths = ['data/enem/microdata_20' + str(i) + '/DADOS/MICRODADOS_ENEM_20' + str(i) + '.csv' for i in range(23, 24)]

def read_csvs(paths):
    dfs = []
    for path in paths:
        try:
            print(f"Lendo arquivo: {path}")
            df = pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)
            dfs.append(df)
        except FileNotFoundError:
            print(f"File not found: {path}")
    return dfs

def transform_cursos_data(dfs):
    print("Transformando dados...")
    filtered = []
    for df in dfs:
        if 'SG_UF' in df.columns:
            filtered.append(df[df['SG_UF'] == 'SP'])
        elif 'SG_UF_IES' in df.columns:
            filtered.append(df[df['SG_UF_IES'] == 'SP'])
        else:
            print("Coluna de UF não encontrada em um dos DataFrames, pulando...")
    print("Transformação concluída")
    return pd.concat(filtered, ignore_index=True)

def main():
    print("Iniciando leitura dos arquivos...")
    cursos_dfs = read_csvs(csv_cursos_paths)
    cursos_dfs = pd.concat(cursos_dfs, ignore_index=True)
    cursos_dfs.to_csv('data/transformed_data/cursos.csv', index=False, sep=';')
    ies_dfs = read_csvs(csv_IES_paths)
    ies_dfs = pd.concat(ies_dfs, ignore_index=True)
    ies_dfs.to_csv('data/transformed_data/ies.csv', index=False, sep=';')
    enem_dfs = read_csvs(csv_enem_paths)
    enem_dfs = pd.concat(enem_dfs, ignore_index=True)
    enem_dfs.to_csv('data/transformed_data/enem.csv', index=False, sep=';')

    '''print("Escrevendo dados transformados...")
    cursos_sp = transform_cursos_data(cursos_dfs)
    cursos_sp.to_csv('data/transformed_data/cursos_sp.csv', index=False, sep=';')    
    ies_sp = transform_cursos_data(ies_dfs)
    ies_sp.to_csv('data/transformed_data/ies_sp.csv', index=False, sep=';')'''

if __name__ == "__main__":
    main()
    print("Processamento concluído.")