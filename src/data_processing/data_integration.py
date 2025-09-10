import pandas as pd
import numpy as np

def integrate_data(cursos_df, ies_df):
    """
    Integra os dataframes de cursos e IES, e cria a variável-alvo.
    O dataframe do ENEM não será usado nesta versão devido à falta de uma chave de junção confiável.
    """
    print("--- Iniciando Integração dos Dados (Foco no Censo) ---")

    # Passo 1: Unir informações de Cursos e IES
    print("Unindo dados de Cursos e IES...")
    # Removemos colunas duplicadas ou desnecessárias de 'ies_df' antes do merge
    ies_df_simplified = ies_df.drop(columns=['TP_CATEGORIA_ADMINISTRATIVA', 'NO_REGIAO_IES', 'SG_UF_IES', 'NO_MUNICIPIO_IES'])
    cursos_ies_df = pd.merge(cursos_df, ies_df_simplified, on='CO_IES', how='left')

    # Passo 2: Engenharia da Variável-Alvo (Target)
    matriculados = cursos_ies_df['QT_MAT'].replace(0, np.nan)
    desvinculados = cursos_ies_df['QT_SIT_DESVINCULADO']
    
    cursos_ies_df['TAXA_EVASAO'] = (desvinculados / matriculados).fillna(0)
    
    mediana_evasao = cursos_ies_df['TAXA_EVASAO'].median()
    cursos_ies_df['ALTA_EVASAO'] = (cursos_ies_df['TAXA_EVASAO'] > mediana_evasao).astype(int)
    print(f"Variável-alvo 'ALTA_EVASAO' criada. Mediana da taxa de evasão: {mediana_evasao:.2f}")

    # Remove colunas que não serão usadas como features no modelo
    final_df = cursos_ies_df.drop(columns=['QT_MAT', 'QT_CONC', 'QT_SIT_DESVINCULADO', 'TAXA_EVASAO'])

    print(f"Integração concluída. Formato do DataFrame final: {final_df.shape}")
    print("--- Fim da Integração ---")
    
    return final_df