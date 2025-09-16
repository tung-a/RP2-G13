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
        'co_ies', 'tp_categoria_administrativa', 'no_regiao_ies'
    ]
    dtype_map = {'co_ies': 'float64', 'co_curso': 'float64'}

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

    # Padronizar e harmonizar tipos de dados
    cursos_df.columns = [col.lower() for col in cursos_df.columns]
    print("--- Harmonizando tipos de dados das chaves de junção... ---")
    for df in [cursos_df, ies_df]:
        if 'co_ies' in df.columns:
            df['co_ies'] = pd.to_numeric(df['co_ies'], errors='coerce').astype('float64')
    if 'co_curso' in cursos_df.columns:
        cursos_df['co_curso'] = pd.to_numeric(cursos_df['co_curso'], errors='coerce').astype('float64')

    # Merge das tabelas pequenas
    cursos_ies_df = pd.merge(cursos_df, ies_df, on='co_ies', how='inner').dropna()

    # Feature Engineering
    alunos_df['tempo_permanencia'] = alunos_df['nu_ano_censo'] - alunos_df['nu_ano_ingresso']
    alunos_df = alunos_df[alunos_df['tp_situacao'] == 2]
    alunos_df = alunos_df[(alunos_df['tempo_permanencia'] > 0) & (alunos_df['tempo_permanencia'] < 20)]

    # --- CORREÇÃO FINAL: AMOSTRAGEM AGRESSIVA ANTES DO MERGE ---
    # Reduzimos o maior dataframe para 2% do seu tamanho original.
    print("--- O dataset de alunos é muito grande. Extraindo uma amostra de 2% ANTES do merge... ---")
    alunos_df = alunos_df.sample(frac=0.02, random_state=42)

    # Merge final com Dask
    print("--- Iniciando merge final com Dask (em dados drasticamente reduzidos)... ---")
    final_df_dask = dd.merge(alunos_df, cursos_ies_df, on=['co_curso', 'co_ies'], how='inner')

    # Limpeza
    final_df_dask = final_df_dask.drop(columns=['nu_ano_censo', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso'])

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