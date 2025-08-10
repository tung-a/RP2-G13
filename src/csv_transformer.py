import pandas as pd

csv_cursos_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_CURSOS_20' + str(i) + '.csv' for i in range(19, 24)]
csv_IESs_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_IES_20' + str(i) + '.csv' for i in range(19, 24)]

def read_csvs(paths):
    dfs = []
    for path in paths:
        try:
            print(f"Lendo arquivo: {path}")
            df = pd.read_csv(path, sep=';', encoding='latin1')
            dfs.append(df)
        except FileNotFoundError:
            print(f"File not found: {path}")
    transform_cursos_data(dfs)
    return dfs

def transform_cursos_data(dfs):
    print("Transformando dados...")
    for df in dfs:
        df = df[df['SG_UF'] == 'SP']
    return pd.concat(dfs, ignore_index=True)

def main():
    print("Iniciando leitura dos arquivos...")
    cursos_dfs = read_csvs(csv_cursos_paths)
    ies_dfs = read_csvs(csv_IESs_paths)

    print("Escrevendo dados transformados...")
    cursos_dfs.to_csv('data/transformed_data/cursos_sp.csv', index=False, sep=';')
    ies_dfs.to_csv('data/transformed_data/ies_sp.csv', index=False, sep=';')