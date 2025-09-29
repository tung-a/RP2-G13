import dask.dataframe as dd
import pandas as pd
import os
import gc

def load_and_integrate_data(data_path):
    """
    Carrega e integra os dados, corrigindo o formato decimal das colunas de
    duração do curso para garantir a criação correta do 'período ideal'.
    """
    # 1. Definir as colunas, incluindo as de tempo de integralização
    aluno_cols = [
        'nu_ano_censo', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso',
        'tp_cor_raca', 'tp_sexo', 'faixa_etaria', 'tp_escola_conclusao_ens_medio',
        'in_apoio_social', 'in_financiamento_estudantil'
    ]
    curso_cols = [
        'NU_ANO_CENSO', 'CO_IES', 'CO_CURSO', 'TP_MODALIDADE_ENSINO', 'NU_CARGA_HORARIA', 
        'TP_GRAU_ACADEMICO', 'NU_INTEGRALIZACAO_INTEGRAL', 'NU_INTEGRALIZACAO_MATUTINO',
        'NU_INTEGRALIZACAO_VESPERTINO', 'NU_INTEGRALIZACAO_NOTURNO', 'NU_INTEGRALIZACAO_EAD'
    ]
    ies_cols = ['co_ies', 'nu_ano_censo', 'tp_categoria_administrativa', 'no_regiao_ies']
    igc_cols = ['ano', 'cod_ies', 'igc', 'igc_fx']
    dtype_map = {'co_ies': 'float64', 'co_curso': 'float64', 'nu_ano_censo': 'float64'}

    # 2. Carregar os dados
    print("--- Carregando arquivos CSV... ---")
    alunos_dd = dd.read_csv(
        os.path.join(data_path, 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_*', '*.csv'),
        sep=';', encoding='latin1', usecols=aluno_cols, dtype=dtype_map, assume_missing=True
    ).dropna()
    cursos_df = pd.read_csv(os.path.join(data_path, 'ces', 'SoU_censo_cursos', 'SoU_censo_curso.csv'), sep=';', encoding='latin1', usecols=curso_cols)
    ies_df = pd.read_csv(os.path.join(data_path, 'ces', 'SoU_censo_IES', 'SoU_censo_IES.csv'), sep=';', encoding='latin1', usecols=ies_cols)
    igc_df = pd.read_csv(os.path.join(data_path, 'IGC', 'igc_tratado.csv'), sep=';', encoding='latin1', usecols=igc_cols)
    igc_df = igc_df.rename(columns={'ano': 'nu_ano_censo', 'cod_ies': 'co_ies'})

    # 3. Criar 'tempo_permanencia' e 'duracao_ideal_anos' (corrigindo o formato decimal)
    alunos_dd['tempo_permanencia'] = alunos_dd['nu_ano_censo'] - alunos_dd['nu_ano_ingresso']
    
    integralizacao_cols = [
        'NU_INTEGRALIZACAO_INTEGRAL', 'NU_INTEGRALIZACAO_MATUTINO',
        'NU_INTEGRALIZACAO_VESPERTINO', 'NU_INTEGRALIZACAO_NOTURNO', 'NU_INTEGRALIZACAO_EAD'
    ]
    
    # --- CORREÇÃO PRINCIPAL AQUI ---
    for col in integralizacao_cols:
        # Substitui a vírgula pelo ponto e converte para número
        cursos_df[col] = cursos_df[col].str.replace(',', '.', regex=False)
        cursos_df[col] = pd.to_numeric(cursos_df[col], errors='coerce')
        
    cursos_df['duracao_ideal_anos'] = cursos_df[integralizacao_cols].bfill(axis=1).iloc[:, 0]
    cursos_df.dropna(subset=['duracao_ideal_anos'], inplace=True)
    cursos_df.drop(columns=integralizacao_cols, inplace=True)

    # 4. Harmonizar e Juntar os Dados
    cursos_df.columns = [col.lower() for col in cursos_df.columns]
    for df in [cursos_df, ies_df, igc_df]:
        for col in ['co_ies', 'nu_ano_censo', 'co_curso']:
            if col in df.columns:
                df[col] = pd.to_numeric(df[col], errors='coerce').astype('Int64')

    # Remove duplicados de cursos, mantendo a informação mais recente
    cursos_df = cursos_df.sort_values('nu_ano_censo', ascending=False).drop_duplicates(subset=['co_ies', 'co_curso'])

    cursos_ies_df = pd.merge(cursos_df, ies_df, on=['co_ies', 'nu_ano_censo'], how='inner')
    cursos_ies_df = pd.merge(cursos_ies_df, igc_df, on=['co_ies', 'nu_ano_censo'], how='left')
    cursos_ies_df['igc'] = cursos_ies_df['igc'].fillna(cursos_ies_df['igc'].mean())

    # 5. Filtragem e Amostragem
    alunos_dd = alunos_dd[alunos_dd['tp_situacao'] == 2]
    alunos_dd = alunos_dd[(alunos_dd['tempo_permanencia'] > 0) & (alunos_dd['tempo_permanencia'] < 20)]
    print("--- Extraindo amostra de 0.5%... ---")
    alunos_dd = alunos_dd.sample(frac=0.005, random_state=42)

    # 6. Merge Final e Limpeza
    print("--- Iniciando merge final com Dask... ---")
    final_df_dask = dd.merge(alunos_dd, cursos_ies_df.drop(columns=['nu_ano_censo']), on=['co_curso', 'co_ies'], how='inner')
    
    cols_to_drop = ['nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso', 'igc_fx']
    final_df_dask = final_df_dask.drop(columns=[col for col in cols_to_drop if col in final_df_dask.columns])

    print("--- Computando o DataFrame final... ---")
    final_df_pandas = final_df_dask.compute()
    final_df_pandas.dropna(inplace=True)

    del final_df_dask, cursos_ies_df, igc_df, ies_df, cursos_df, alunos_dd
    gc.collect()

    print("Dados integrados e limpos com sucesso.")
    print(f"Total de registros para modelagem: {len(final_df_pandas)}")
    print("Colunas finais:", final_df_pandas.columns.tolist())

    return final_df_pandas

def split_by_institution_type(df):
    if 'tp_categoria_administrativa' not in df.columns:
        return pd.DataFrame(), pd.DataFrame()
    df_publica = df[df['tp_categoria_administrativa'].isin([1, 2, 3])].copy()
    df_privada = df[df['tp_categoria_administrativa'].isin([4, 5, 6, 7, 8])].copy()
    return df_publica, df_privada