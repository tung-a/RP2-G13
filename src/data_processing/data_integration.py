import pandas as pd
import numpy as np

def integrate_data(cursos_df, ies_df):
    """
    Integra os dataframes de cursos e IES, preservando as colunas
    necessárias para a análise e criando a variável-alvo.
    """
    print("--- Iniciando Integração dos Dados ---")

    # Garante que a coluna necessária para a análise ('TP_CATEGORIA_ADMINISTRATIVA')
    # que vem do dataframe de IES, seja mantida após a junção.
    cursos_ies_df = pd.merge(cursos_df, ies_df, on='CO_IES', how='left')

    # Checagem de segurança para garantir que a coluna está presente
    if 'TP_CATEGORIA_ADMINISTRATIVA' not in cursos_ies_df.columns:
        raise ValueError("ERRO CRÍTICO: A coluna 'TP_CATEGORIA_ADMINISTRATIVA' não foi encontrada após o merge.")

    # Engenharia da Variável-Alvo (Target)
    matriculados = cursos_ies_df['QT_MAT'].replace(0, np.nan)
    desvinculados = cursos_ies_df['QT_SIT_DESVINCULADO']
    
    cursos_ies_df['TAXA_EVASAO'] = (desvinculados / matriculados).fillna(0)
    
    mediana_evasao = cursos_ies_df['TAXA_EVASAO'].median()
    cursos_ies_df['ALTA_EVASAO'] = (cursos_ies_df['TAXA_EVASAO'] > mediana_evasao).astype(int)
    print(f"Variável-alvo 'ALTA_EVASAO' criada. Mediana da taxa de evasão: {mediana_evasao:.2f}")

    print(f"Integração concluída. Formato do DataFrame final: {cursos_ies_df.shape}")
    print("--- Fim da Integração ---")
    
    return cursos_ies_df