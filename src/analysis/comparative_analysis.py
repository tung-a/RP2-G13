import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os

def run_ideal_time_analysis(df, report_path='reports/figures'):
    """
    Compara o tempo de permanência real com a duração ideal do curso,
    cria categorias e gera gráficos e análises.
    """
    if 'duracao_ideal_anos' not in df.columns or 'tempo_permanencia' not in df.columns:
        print("AVISO: As colunas 'duracao_ideal_anos' ou 'tempo_permanencia' não foram encontradas para a análise comparativa.")
        return df

    print("\n--- INICIANDO ANÁLISE COMPARATIVA COM O TEMPO IDEAL DO CURSO ---")
    
    # 1. Calcular a diferença em anos
    df['diferenca_permanencia'] = df['tempo_permanencia'] - df['duracao_ideal_anos']

    # 2. Criar categorias para o status do aluno
    bins = [-np.inf, -0.5, 0.5, np.inf]
    labels = ['Evasão Provável', 'Conclusão no Prazo', 'Atraso']
    df['status_conclusao'] = pd.cut(df['diferenca_permanencia'], bins=bins, labels=labels)

    # Cria a pasta para salvar os relatórios se não existir
    os.makedirs(report_path, exist_ok=True)

    # 3. Gerar e salvar gráfico de distribuição geral
    plt.figure(figsize=(10, 6))
    sns.countplot(x='status_conclusao', data=df, order=labels, palette='viridis')
    plt.title('Distribuição Geral de Alunos por Status de Conclusão', fontsize=16)
    plt.xlabel('Status de Conclusão', fontsize=12)
    plt.ylabel('Número de Alunos', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, 'distribuicao_status_conclusao.png'))
    plt.close()

    # 4. Gerar e salvar gráfico comparativo entre IES Públicas e Privadas
    df['tipo_ies'] = df['tp_categoria_administrativa'].apply(lambda x: 'Pública' if x in [1, 2, 3] else 'Privada')
    
    plt.figure(figsize=(12, 7))
    sns.countplot(x='status_conclusao', hue='tipo_ies', data=df, order=labels, palette='mako')
    plt.title('Status de Conclusão vs. Tipo de Instituição (Pública vs. Privada)', fontsize=16)
    plt.xlabel('Status de Conclusão', fontsize=12)
    plt.ylabel('Número de Alunos', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(report_path, 'comparativo_status_por_tipo_ies.png'))
    plt.close()

    print("-> Gráficos da análise comparativa foram salvos em 'reports/figures/'.")

    return df