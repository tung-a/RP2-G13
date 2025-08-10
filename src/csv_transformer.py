import pandas as pd

csv_cursos_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_CURSOS_20' + str(i) + '.csv' for i in range(19, 24)]
csv_IESs_paths = ['data/ces/microdata_20' + str(i) + '/dados/MICRODADOS_CADASTRO_IES_20' + str(i) + '.csv' for i in range(19, 24)]

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
    ies_dfs = read_csvs(csv_IESs_paths)

    print("Escrevendo dados transformados...")
    cursos_sp = transform_cursos_data(cursos_dfs)
    cursos_sp.to_csv('data/transformed_data/cursos_sp.csv', index=False, sep=';')    
    ies_sp = transform_cursos_data(ies_dfs)
    ies_sp.to_csv('data/transformed_data/ies_sp.csv', index=False, sep=';')

if __name__ == "__main__":
    main()
    print("Processamento concluído.")