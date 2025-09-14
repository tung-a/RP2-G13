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
    tipos_ies = {
        'Pública': [1, 2, 3],
        'Privada': [4, 5]
    }
    data['TIPO_IES'] = data['TP_CATEGORIA_ADMINISTRATIVA'].apply(
        lambda x: 'Pública' if x in tipos_ies['Pública'] else 'Privada'
    )

    # 2. Análise Estatística
    print("\n--- Análise Estatística da Taxa de Evasão ---")
    matriculados = data['QT_MAT'].replace(0, pd.NA)
    desvinculados = data['QT_SIT_DESVINCULADO']
    data['TAXA_EVASAO'] = (desvinculados / matriculados).dropna()

    evasao_stats = data.groupby('TIPO_IES')['TAXA_EVASAO'].describe()
    print("\nEstatísticas da Taxa de Evasão por Tipo de IES:")
    print(evasao_stats)
    
    evasao_stats.to_csv('reports/estatisticas_evasao.csv')
    print("\nEstatísticas salvas em 'reports/estatisticas_evasao.csv'")

    # 3. Geração de Visualizações
    print("\n--- Gerando Visualizações ---")
    
    # Gráfico 1: Boxplot da Taxa de Evasão (sem alterações)
    plt.figure(figsize=(10, 6))
    sns.boxplot(x='TIPO_IES', y='TAXA_EVASAO', data=data, showfliers=False) # Remove outliers para melhor visualização
    plt.title('Distribuição da Taxa de Evasão por Tipo de IES (sem outliers)')
    plt.ylabel('Taxa de Evasão')
    plt.xlabel('Tipo de Instituição')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    boxplot_path = 'reports/boxplot_taxa_evasao.png'
    plt.savefig(boxplot_path)
    print(f"Gráfico de boxplot salvo em: {boxplot_path}")
    plt.close()

    # Gráfico 2: Histograma da Taxa de Evasão (VERSÃO MELHORADA)
    # Filtra dados para focar no intervalo de 0 a 1 (0% a 100% de evasão)
    plot_data = data[(data['TAXA_EVASAO'] >= 0) & (data['TAXA_EVASAO'] <= 1)].copy()

    # Cria subplots separados para cada tipo de IES
    g = sns.FacetGrid(plot_data, col="TIPO_IES", height=6, aspect=1.2, col_wrap=2)
    
    # Mapeia o histograma para cada subplot, com escala de log na frequência (eixo Y)
    g.map(sns.histplot, "TAXA_EVASAO", kde=True, bins=50, log_scale=(False, True))
    
    g.fig.suptitle('Distribuição da Taxa de Evasão (Frequência em Escala Logarítmica)', y=1.03, fontsize=16)
    g.set_axis_labels("Taxa de Evasão", "Frequência (em escala log)")
    g.set_titles("IES do Tipo: {col_name}")
    plt.tight_layout()

    hist_path_melhorado = 'reports/histograma_evasao_separado.png'
    plt.savefig(hist_path_melhorado)
    print(f"Gráfico de histograma melhorado salvo em: {hist_path_melhorado}")
    plt.close()

    print("\n--- Análise Comparativa Concluída ---")