import dask.dataframe as dd
import pandas as pd
import numpy as np
import os
import gc

def load_and_integrate_data(data_path, nivel_especifico_categoria:bool = True):
    """
    Carrega e integra os dados, corrigindo o formato decimal das colunas de
    duração do curso para garantir a criação correta do 'período ideal'.
    """

    # A coluna categórica do curso pode variar entre CO_CINE_AREA_GERAL	e CO_CINE_AREA_ESPECIFICA

    # 1. Definir as colunas, incluindo as de tempo de integralização
    aluno_cols = [
        'nu_ano_censo', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso',
        'tp_cor_raca', 'tp_sexo', 'faixa_etaria', 'tp_escola_conclusao_ens_medio',
        'in_apoio_social', 'in_financiamento_estudantil'
    ]
    curso_cols = [
        'NU_ANO_CENSO', 'CO_IES', 'CO_CURSO', 'TP_MODALIDADE_ENSINO', 'NU_CARGA_HORARIA', 
        'TP_GRAU_ACADEMICO', 'NU_INTEGRALIZACAO_INTEGRAL', 'NU_INTEGRALIZACAO_MATUTINO', 
        'NU_INTEGRALIZACAO_VESPERTINO', 'NU_INTEGRALIZACAO_NOTURNO', 'NU_INTEGRALIZACAO_EAD',
        'QT_INSCRITO_TOTAL','QT_VAGA_TOTAL','CO_CINE_ROTULO','SIGLA_UF_CURSO'
    ]
    ies_cols = ['co_ies', 'nu_ano_censo', 'tp_categoria_administrativa', 'no_regiao_ies']
    igc_cols = ['ano', 'cod_ies', 'igc', 'igc_fx']
    dtype_map = {'co_ies': 'float64', 'co_curso': 'float64', 'nu_ano_censo': 'float64'}

    cine_cols = ["CO_CINE_AREA_GERAL", "NM_CINE_AREA_GERAL", "CO_CINE_AREA_ESPECIFICA", "NM_CINE_AREA_ESPECIFICA", "CO_CINE_ROTULO"]

    pib_cols = ['Sigla', 'UF', '2014', '2015', '2016', '2017', '2018', '2019', '2020', '2021', '2022']

    # 2. Carregar os dados
    print("--- Carregando arquivos CSV... ---")
    alunos_dd = dd.read_csv(
        os.path.join(data_path, 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_*', '*.csv'),
        sep=';', encoding='latin1', usecols=aluno_cols, dtype=dtype_map, assume_missing=True
    ).dropna()
    cursos_df = pd.read_csv(os.path.join(data_path, 'ces', 'SoU_censo_cursos', 'SoU_censo_curso.csv'), sep=';', encoding='latin1', usecols=curso_cols, low_memory=False)
    ies_df = pd.read_csv(os.path.join(data_path, 'ces', 'SoU_censo_IES', 'SoU_censo_IES.csv'), sep=';', encoding='latin1', usecols=ies_cols)
    igc_df = pd.read_csv(os.path.join(data_path, 'igc', 'igc_tratado.csv'), sep=';', encoding='latin1', usecols=igc_cols)
    cine_df = pd.read_csv(os.path.join(data_path, 'cine', 'cine.csv'), sep=',', encoding='utf8', usecols=cine_cols)
    pib_df = pd.read_csv(os.path.join(data_path, 'ibge', 'pib_tratado.csv'), sep=';', encoding='latin1', usecols=pib_cols)

    igc_df = igc_df.rename(columns={'ano': 'nu_ano_censo', 'cod_ies': 'co_ies'})

    print(pib_df)
    # Preparar o DataFrame Pib para merge futuro
    id_vars = ['Sigla', 'UF']
    colunas_pib = [str(col) for col in pib_df.columns if col not in id_vars]

    pib_df = pib_df.melt(
        id_vars=id_vars,
        value_vars=colunas_pib,
        var_name='ano',
        value_name='pib'
    )
    pib_df['ano'] = pib_df['ano'].astype('Int64')
    pib_df = pib_df.drop(columns=['UF'])

    if nivel_especifico_categoria:
        cine_df = cine_df.rename(columns={'NM_CINE_AREA_ESPECIFICA': 'nm_categoria', 'NM_CINE_AREA_GERAL': 'nm_categoria_dropar'})
    else:
        cine_df = cine_df.rename(columns={'NM_CINE_AREA_GERAL': 'nm_categoria', 'NM_CINE_AREA_ESPECIFICA': 'nm_categoria_dropar'})

    cine_df = cine_df.drop(columns=['nm_categoria_dropar','CO_CINE_AREA_GERAL','CO_CINE_AREA_ESPECIFICA'])
    
    num_linhas = cursos_df.shape[0]
    cursos_df = pd.merge(cursos_df, cine_df, on='CO_CINE_ROTULO', how='inner')
    print(f"Quantidade de cursos antes do merge: {num_linhas}. Quantidade de cursos após o merge com CINE: {cursos_df.shape[0]}.")

    print("--- Iniciando merge com PIB... ---")
    cursos_df['NU_ANO_CENSO'] = pd.to_numeric(cursos_df['NU_ANO_CENSO'], errors='coerce').astype('Int64')
    print(cursos_df.dtypes)
    num_linhas = cursos_df.shape[0]
    cursos_df = pd.merge(cursos_df, pib_df, left_on=['SIGLA_UF_CURSO','NU_ANO_CENSO'], right_on=['Sigla','ano'], how='inner')
    cursos_df = cursos_df.drop(columns=['Sigla','ano'])
    print(f"Quantidade de cursos antes do merge: {num_linhas}. Quantidade de cursos após o merge com ibge: {cursos_df.shape[0]}.")
    print(cursos_df)
    print(cursos_df.dtypes)

    # 3. Criar 'tempo_permanencia' e 'duracao_ideal_anos' (corrigindo o formato decimal)
    alunos_dd['tempo_permanencia'] = alunos_dd['nu_ano_censo'] - alunos_dd['nu_ano_ingresso']
    
    integralizacao_cols = [
        'NU_INTEGRALIZACAO_INTEGRAL', 'NU_INTEGRALIZACAO_MATUTINO',
        'NU_INTEGRALIZACAO_VESPERTINO', 'NU_INTEGRALIZACAO_NOTURNO', 'NU_INTEGRALIZACAO_EAD'
    ]
    cursos_df["inscritos_por_vaga"] = np.divide(
        cursos_df["QT_INSCRITO_TOTAL"],
        cursos_df["QT_VAGA_TOTAL"],
        out=np.full_like(cursos_df["QT_INSCRITO_TOTAL"], 0.0, dtype=float), 
        where=cursos_df["QT_VAGA_TOTAL"] != 0
    )
    cursos_df = cursos_df.drop(columns=["QT_INSCRITO_TOTAL","QT_VAGA_TOTAL"])

    # --- CORREÇÃO PRINCIPAL AQUI ---
    for col in integralizacao_cols:
        # Substitui a vírgula pelo ponto e converte para número
        cursos_df[col] = cursos_df[col].str.replace(',', '.', regex=False)
        cursos_df[col] = pd.to_numeric(cursos_df[col], errors='coerce')
        
    cursos_df['duracao_ideal_anos'] = cursos_df[integralizacao_cols].mean(axis=1)
    cursos_df.dropna(subset=['duracao_ideal_anos'], inplace=True)
    cursos_df.drop(columns=integralizacao_cols, inplace=True)

    # 4. Harmonizar e Juntar os Dados
    cursos_df.columns = [col.lower() for col in cursos_df.columns]
    for df in [alunos_dd, cursos_df, ies_df, igc_df]:
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
    
    final_df_dask['taxa_integralizacao'] = final_df_dask['tempo_permanencia'] / final_df_dask['duracao_ideal_anos']

    cols_to_drop = ['tempo_permanencia', 'nu_ano_ingresso', 'tp_situacao', 'co_ies', 'co_curso', 'igc_fx','co_cine_rotulo']
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