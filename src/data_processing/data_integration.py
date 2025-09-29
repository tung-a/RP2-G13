import dask.dataframe as dd
import pandas as pd
import os
import gc

def load_and_integrate_data(data_path):
    """
    Carrega e integra os dados usando Dask, aplicando uma amostragem agressiva
    no maior dataset ANTES do merge para garantir que o processo caiba na memória.
    """
    # Definir colunas e tipos de dados
    aluno_cols = [
        'nu_ano_censo', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso',
        'tp_cor_raca', 'tp_sexo', 'faixa_etaria', 'tp_escola_conclusao_ens_medio',
        'in_apoio_social', 'in_financiamento_estudantil'
    ]
    curso_cols = [
        'CO_IES', 'CO_CURSO', 'TP_MODALIDADE_ENSINO', 'NU_CARGA_HORARIA', 'TP_GRAU_ACADEMICO'
    ]
    ies_cols = [
        'co_ies','nu_ano_censo', 'tp_categoria_administrativa', 'no_regiao_ies'
    ]
    igc_cols = ['co_ies', 'nu_ano_censo', 'igc', 'igc_fx']
    
    dtype_map = {'co_ies': 'float64', 'co_curso': 'float64', 'nu_ano_censo': 'float64'}

    # Carregar dados com Dask
    print("--- Carregando arquivos CSV com Dask... ---")
    alunos_df = dd.read_csv(
        os.path.join(data_path, 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_*', '*.csv'),
        sep=';', encoding='latin1', usecols=aluno_cols, dtype=dtype_map, assume_missing=True
    )

    # Carregar dados de cursos e IES
    cursos_path = os.path.join(data_path, 'ces', 'SoU_censo_cursos', 'SoU_censo_curso.csv')
    ies_path = os.path.join(data_path, 'ces', 'SoU_censo_IES', 'SoU_censo_IES.csv')
    cursos_df = pd.read_csv(cursos_path, sep=';', encoding='latin1', usecols=curso_cols)
    ies_df = pd.read_csv(ies_path, sep=';', encoding='latin1', usecols=ies_cols)

    # NOVO PASSO: Carregar os dados do IGC
    print("--- Carregando dados do IGC... ---")
    igc_path = os.path.join(data_path, 'IGC', 'igc_tratado.csv')
    igc_df = pd.read_csv(igc_path, sep=';', encoding='latin1', usecols=igc_cols)

    print(igc_df.head())

    # Padronizar e harmonizar tipos de dados
    cursos_df.columns = [col.lower() for col in cursos_df.columns]
    print("--- Harmonizando tipos de dados das chaves de junção... ---")
    for df in [cursos_df, ies_df,igc_df]:
        if 'co_ies' in df.columns:
            df['co_ies'] = pd.to_numeric(df['co_ies'], errors='coerce').astype('float64')
        if 'nu_ano_censo' in df.columns:
            df['nu_ano_censo'] = pd.to_numeric(df['nu_ano_censo'], errors='coerce').astype('float64')
        if 'co_curso' in df.columns:
            df['co_curso'] = pd.to_numeric(df['co_curso'], errors='coerce').astype('float64')

    # Merge das tabelas pequenas
    cursos_ies_df = pd.merge(cursos_df, ies_df, on='co_ies', how='inner').dropna()
    cursos_ies_df = pd.merge(cursos_ies_df, igc_df, on=['co_ies', 'nu_ano_censo'], how='left')
    cursos_ies_df['igc'] = cursos_ies_df['igc'].fillna(2.6278)
    cursos_ies_df['igc_fx'] = cursos_ies_df['igc_fx'].fillna(3)

    # Feature Engineering
    alunos_df['tempo_permanencia'] = alunos_df['nu_ano_censo'] - alunos_df['nu_ano_ingresso']
    alunos_df = alunos_df[alunos_df['tp_situacao'] == 2]
    alunos_df = alunos_df[(alunos_df['tempo_permanencia'] > 0) & (alunos_df['tempo_permanencia'] < 20)]

    # --- CORREÇÃO FINAL: AMOSTRAGEM AGRESSIVA ANTES DO MERGE ---
    # Reduzimos o maior dataframe para 2% do seu tamanho original.
    print("--- O dataset de alunos é muito grande. Extraindo uma amostra de 0.5% ANTES do merge... ---")
    alunos_df = alunos_df.sample(frac=0.005, random_state=42)

    # Merge final com Dask
    print("--- Iniciando merge final com Dask (em dados drasticamente reduzidos)... ---")
    final_df_dask = dd.merge(alunos_df, cursos_ies_df, on=['co_curso', 'co_ies'], how='inner')

    # Renomeando o DataFrame
    final_df_dask = final_df_dask.rename(columns={'nu_ano_censo_y': 'ano_censo_aluno'})
    print(final_df_dask.head())
    print(final_df_dask.columns)

    # Limpeza
    final_df_dask = final_df_dask.drop(columns=['nu_ano_censo_x', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso','igc_fx'])

    print("Grafo de tarefas Dask construído. Iniciando computação final...")
    final_df_pandas = final_df_dask.compute()

    del final_df_dask
    gc.collect()

    print("Dados integrados e limpos com sucesso.")
    print(f"Total de registros para modelagem: {len(final_df_pandas)}")

    return final_df_pandas

def split_by_institution_type(df):
    if 'tp_categoria_administrativa' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df_publica = df[df['tp_categoria_administrativa'].isin([1, 2, 3])].copy()
    df_privada = df[df['tp_categoria_administrativa'].isin([4, 5, 6, 7, 8])].copy()
    return df_publica, df_privada


def load_and_integrate_data2(data_path):
    """
    Carrega e integra os dados usando Dask, aplicando uma amostragem agressiva
    no maior dataset ANTES do merge para garantir que o processo caiba na memória.
    """
    # Definir colunas e tipos de dados
    aluno_cols = [
        'nu_ano_censo', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso',
        'tp_cor_raca', 'tp_sexo', 'faixa_etaria', 'tp_escola_conclusao_ens_medio',
        'in_apoio_social', 'in_financiamento_estudantil'
    ]
    curso_cols = [
        'CO_IES', 'CO_CURSO', 'TP_MODALIDADE_ENSINO', 'NU_CARGA_HORARIA', 'TP_GRAU_ACADEMICO'
    ]
    ies_cols = [
        'co_ies','nu_ano_censo', 'tp_categoria_administrativa', 'no_regiao_ies'
    ]
    igc_cols = ['co_ies', 'nu_ano_censo', 'igc', 'igc_fx']
    
    dtype_map = {'co_ies': 'float64', 'co_curso': 'float64', 'nu_ano_censo': 'float64'}

    # Carregar dados com Dask
    print("--- Carregando arquivos CSV com Dask... ---")
    alunos_df = dd.read_csv(
        os.path.join(data_path, 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_*', '*.csv'),
        sep=';', encoding='latin1', usecols=aluno_cols, dtype=dtype_map, assume_missing=True
    )

    # Carregar dados de cursos e IES
    cursos_path = os.path.join(data_path, 'ces', 'SoU_censo_cursos', 'SoU_censo_curso.csv')
    ies_path = os.path.join(data_path, 'ces', 'SoU_censo_IES', 'SoU_censo_IES.csv')
    cursos_df = pd.read_csv(cursos_path, sep=';', encoding='latin1', usecols=curso_cols)
    ies_df = pd.read_csv(ies_path, sep=';', encoding='latin1', usecols=ies_cols)

    print("--- Carregando dados do IGC... ---")
    igc_path = os.path.join(data_path, 'IGC', 'igc_tratado.csv')
    igc_df = pd.read_csv(igc_path, sep=';', encoding='latin1', usecols=igc_cols)

    print(igc_df.head())

    # Padronizar e harmonizar tipos de dados
    cursos_df.columns = [col.lower() for col in cursos_df.columns]
    print("--- Harmonizando tipos de dados das chaves de junção... ---")
    for df in [cursos_df, ies_df,igc_df]:
        if 'co_ies' in df.columns:
            df['co_ies'] = pd.to_numeric(df['co_ies'], errors='coerce').astype('float64')
        if 'nu_ano_censo' in df.columns:
            df['nu_ano_censo'] = pd.to_numeric(df['nu_ano_censo'], errors='coerce').astype('float64')
        if 'co_curso' in df.columns:
            df['co_curso'] = pd.to_numeric(df['co_curso'], errors='coerce').astype('float64')

    # Merge das tabelas pequenas
    cursos_ies_df = pd.merge(cursos_df, ies_df, on='co_ies', how='inner').dropna()
    cursos_ies_df = pd.merge(cursos_ies_df, igc_df, on=['co_ies', 'nu_ano_censo'], how='left')

    # Feature Engineering
    alunos_df['tempo_permanencia'] = alunos_df['nu_ano_censo'] - alunos_df['nu_ano_ingresso']
    alunos_df = alunos_df[alunos_df['tp_situacao'] == 2]
    alunos_df = alunos_df[(alunos_df['tempo_permanencia'] > 0) & (alunos_df['tempo_permanencia'] < 20)]

    # --- CORREÇÃO FINAL: AMOSTRAGEM AGRESSIVA ANTES DO MERGE ---
    # Reduzimos o maior dataframe para 2% do seu tamanho original.
    print("--- O dataset de alunos é muito grande. Extraindo uma amostra de 0.5% ANTES do merge... ---")
    alunos_df = alunos_df.sample(frac=0.005, random_state=42)

    # Merge final com Dask
    print("--- Iniciando merge final com Dask (em dados drasticamente reduzidos)... ---")

    final_df_dask = dd.merge(alunos_df, igc_df, on=['nu_ano_censo', 'co_ies'], how='inner')
    final_df_dask = final_df_dask.rename(columns={'nu_ano_censo': 'ano_censo_aluno'})
    print(final_df_dask)
    print(final_df_dask.columns)
    final_df_dask = dd.merge(alunos_df, cursos_ies_df, on=['co_curso', 'co_ies'], how='inner')

    print(final_df_dask.head())
    print(final_df_dask.columns)

    # Limpeza
    final_df_dask = final_df_dask.drop(columns=['nu_ano_censo_y', 'nu_ano_censo_x', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso','igc_fx'])

    print("Grafo de tarefas Dask construído. Iniciando computação final...")
    final_df_pandas = final_df_dask.compute()

    del final_df_dask
    gc.collect()

    print("Dados integrados e limpos com sucesso.")
    print(f"Total de registros para modelagem: {len(final_df_pandas)}")

    return final_df_pandas