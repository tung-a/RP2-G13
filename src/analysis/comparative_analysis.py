import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_permanence(data):
    """
    Realiza uma análise comparativa da permanência (evasão) entre
    instituições públicas e privadas.
    """
    print("--- Iniciando Análise Comparativa de Permanência ---")
    os.makedirs('reports', exist_ok=True)

    # 1. Preparação dos Dados
    # Define os tipos de IES com base na coluna 'TP_CATEGORIA_ADMINISTRATIVA'
    tipos_ies = {
        'Pública': [1, 2, 3],
        'Privada': [4, 5]
    }
    
    # Cria uma nova coluna 'TIPO_IES' para facilitar a análise
    data['TIPO_IES'] = data['TP_CATEGORIA_ADMINISTRATIVA'].apply(
        lambda x: 'Pública' if x in tipos_ies['Pública'] else 'Privada'
    )

    # 2. Análise Estatística
    print("\n--- Análise Estatística da Taxa de Evasão ---")
    # Calcula a taxa de evasão
    matriculados = data['QT_MAT'].replace(0, pd.NA)
    desvinculados = data['QT_SIT_DESVINCULADO']
    data['TAXA_EVASAO'] = (desvinculados / matriculados).dropna()

    # Gera estatísticas descritivas por tipo de IES
    evasao_stats = data.groupby('TIPO_IES')['TAXA_EVASAO'].describe()
    print("\nEstatísticas da Taxa de Evasão por Tipo de IES:")
    print(evasao_stats)
    
    # Salva as estatísticas em um arquivo CSV
    evasao_stats.to_csv('reports/estatisticas_evasao.csv')
    print("\nEstatísticas salvas em 'reports/estatisticas_evasao.csv'")

    # 3. Geração de Visualizações
    print("\n--- Gerando Visualizações ---")
    
    # Gráfico 1: Boxplot da Taxa de Evasão
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='TIPO_IES', y='TAXA_EVASAO', data=data)
    plt.title('Distribuição da Taxa de Evasão por Tipo de IES')
    plt.ylabel('Taxa de Evasão')
    plt.xlabel('Tipo de Instituição')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    boxplot_path = 'reports/boxplot_taxa_evasao.png'
    plt.savefig(boxplot_path)
    print(f"Gráfico de boxplot salvo em: {boxplot_path}")
    plt.close()

    # Gráfico 2: Histograma da Taxa de Evasão
    plt.figure(figsize=(12, 7))
    sns.histplot(data=data, x='TAXA_EVASAO', hue='TIPO_IES', kde=True, bins=50)
    plt.title('Histograma da Taxa de Evasão para IES Públicas e Privadas')
    plt.xlabel('Taxa de Evasão')
    plt.ylabel('Frequência')
    
    hist_path = 'reports/histograma_taxa_evasao.png'
    plt.savefig(hist_path)
    print(f"Gráfico de histograma salvo em: {hist_path}")
    plt.close()

    print("\n--- Análise Comparativa Concluída ---")