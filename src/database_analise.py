import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse

def create_comparative_charts(target_column):
    """
    Carrega os dados de IES públicas e privadas e gera gráficos de barras
    comparativos para analisar o impacto médio de diversas características
    em uma coluna-alvo (ex: 'taxa_integralizacao').
    """
    # --- 1. Definição de Caminhos ---
    DATA_PATH = 'data'
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'descriptive_analysis')
    os.makedirs(FIGURES_PATH, exist_ok=True)

    print(f"--- INICIANDO ANÁLISE DESCRITIVA (ALVO: {target_column}) ---")

    # --- 2. Carregar e Combinar os Dados ---
    try:
        df_publica = pd.read_csv(os.path.join(DATA_PATH, 'publica_sample.csv'))
        df_privada = pd.read_csv(os.path.join(DATA_PATH, 'privada_sample.csv'))
    except FileNotFoundError as e:
        print(f"ERRO: Arquivo CSV não encontrado. Verifique o caminho: {e}")
        return

    # Adiciona a coluna de identificação para a legenda do gráfico
    df_publica['Tipo IES'] = 'PUBLICA'
    df_privada['Tipo IES'] = 'PRIVADA'
    
    # Junta os dois DataFrames em um só
    combined_df = pd.concat([df_publica, df_privada], ignore_index=True)
    print(combined_df)
    
    if target_column not in combined_df.columns:
        print(f"ERRO: A coluna-alvo '{target_column}' não existe no CSV. Verifique o nome.")
        return

    # --- 3. Mapeamento de Códigos para Nomes Legíveis (MELHORA OS GRÁFICOS) ---
    # Este passo é essencial para que os eixos dos gráficos sejam compreensíveis.
    map_cor_raca = {
        0: 'Não Declarada', 1: 'Branca', 2: 'Preta', 3: 'Parda',
        4: 'Amarela', 5: 'Indígena', 9: 'Não Dispõe'
    }
    map_sexo = {1: 'Feminino', 2: 'Masculino'}
    map_escola = {1.0: 'Pública', 2.0: 'Privada', 9.0: 'Não Informado' }
    map_grau = {1.0: 'Bacharelado', 2.0: 'Licenciatura', 3.0: 'Tecnológico', 4.0: 'Bacharelado e Licenciatura', 5.0: 'Outros' }
    map_faixa_etaria = {
        1: '1 até 17', 2: '18 - 21', 3: '22 - 25', 4: '26 - 29',
        5: '30 - 33', 6: '34 - 37', 7: '38 - 41', 8: '42 - 45',
        9: '46 - 49', 10: '50 - 53', 11: '54 - 57', 12: '58 - 61',
        13: '62 - 65', 14: 'maior do que 65'
    }

    # Aplica o mapeamento
    combined_df['tp_cor_raca'] = pd.to_numeric(combined_df['tp_cor_raca'], errors='coerce').map(map_cor_raca)
    combined_df['tp_sexo'] = pd.to_numeric(combined_df['tp_sexo'], errors='coerce').map(map_sexo)
    combined_df['tp_escola_conclusao_ens_medio'] = pd.to_numeric(combined_df['tp_escola_conclusao_ens_medio'], errors='coerce').map(map_escola)
    combined_df['tp_grau_academico'] = pd.to_numeric(combined_df['tp_grau_academico'], errors='coerce').map(map_grau)
    combined_df['faixa_etaria'] = pd.to_numeric(combined_df['faixa_etaria'], errors='coerce').map(map_faixa_etaria)

    combined_df['nm_categoria'] = combined_df['nm_categoria'].str.replace(
    'Programas interdisciplinares abrangendo', 
    'Prog. Interdisciplinar', 
    regex=False
)
    combined_df['duracao_ideal_anos'] = pd.to_numeric(combined_df['duracao_ideal_anos'], errors='coerce')
    if 'pib' in combined_df.columns:
        # Cria a versão logarítmica do PIB, que será usada no gráfico de linha
        combined_df['pib_log'] = np.log1p(combined_df['pib'])

# --- 4. Selecionar Features e Gerar Gráficos ---
    # Lista de colunas categóricas para analisar
    features_to_analyze = [
        'tp_cor_raca',
        'tp_sexo',
        'faixa_etaria',
        'tp_escola_conclusao_ens_medio',
        'tp_grau_academico',
        'no_regiao_ies',
        'nm_categoria'
    ]

    print(f"\nIniciando geração de {len(features_to_analyze)} gráficos...")
    
    # Define o título do eixo Y de forma mais amigável
    y_axis_title = target_column.replace('_', ' ').title()

    for feature in features_to_analyze:
        if feature not in combined_df.columns:
            print(f"AVISO: A coluna '{feature}' não foi encontrada. Pulando gráfico.")
            continue
        
        # Agrupa os dados por tipo de IES e pela característica, calculando a média da coluna-alvo
        # Esta é a linha principal que prepara os dados para o gráfico
        combined_impact = combined_df.groupby(['Tipo IES', feature])[target_column].mean().reset_index()

        # Estilo e tamanho da figura
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(14, 8))

        # Cria o gráfico de barras usando Seaborn
        ax = sns.barplot(
            x=feature, 
            y=target_column, 
            hue='Tipo IES', 
            data=combined_impact, 
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'}
        )

        # Títulos e labels
        plt.title(f'Análise da Média de "{y_axis_title}" por "{feature}"', fontsize=16, fontweight='bold')
        plt.ylabel(f'Média de {y_axis_title}', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        
        # Ajustes no eixo X para melhor legibilidade
        plt.xticks(rotation=45, ha='right')
        
        # Adiciona os valores numéricos no topo de cada barra
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}', 
                            ha='center', va='bottom', fontsize=10, color='black')

        # Layout e salvamento do arquivo
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_PATH, f'analise_{target_column}_por_{feature.lower().replace("/", "")}.png')
        plt.savefig(plot_path)
        plt.close() # Fecha a figura para liberar memória
        
        print(f"-> Gráfico para '{feature}' salvo em: {plot_path}")
        
    # --- 5. Selecionar Features Numéricas e Gerar Gráficos de Dispersão (Scatter Plots) ---
    
    numeric_features_to_analyze = [
        'igc',                   # Idade do aluno (ou outra coluna de idade/tempo se existir)
        'pib',                   # Exemplo de nota geral (se existir)
        'inscritos_por_vaga',    # Quantidade de inscritos (se existir)]
        'duracao_ideal_anos'
    ]
    numeric_features_to_analyze = [
        col for col in numeric_features_to_analyze if col in combined_df.columns
    ]

    print(f"\nIniciando geração de {len(numeric_features_to_analyze)} gráficos...")

    # --- LOOP PRINCIPAL CORRIGIDO E UNIFICADO ---

    for feature in numeric_features_to_analyze:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        x_feature = feature
        # Atualiza o título do eixo X, garantindo a capitalização correta
        x_axis_title = feature.replace('_', ' ').title() 

        # Define o número de quantis (bins). 10 é um bom padrão, 5 como fallback.
        num_quantiles = 10 
        
        # Rótulo da coluna de agrupamento temporária
        group_col = f'{feature}_Group'
        
        # Agrupa a feature em faixas (bins) usando qcut (quantis).
        try:
            combined_df[group_col] = pd.qcut(combined_df[x_feature], q=num_quantiles, labels=False, duplicates='drop')
        except ValueError as e:
            # Se 10 quantis não for possível devido a valores repetidos, tenta 5.
            print(f"AVISO: Não foi possível agrupar '{feature}' em {num_quantiles} faixas. Tentando 5 faixas. Erro: {e}")
            num_quantiles = 5
            try:
                combined_df[group_col] = pd.qcut(combined_df[x_feature], q=num_quantiles, labels=False, duplicates='drop')
            except ValueError as e2:
                print(f"ERRO FATAL: Não foi possível agrupar '{feature}' nem com 5 faixas. Pulando este gráfico. Erro: {e2}")
                plt.close() # Fecha a figura aberta
                continue
        
        # Se 'duracao_ideal_anos' foi agrupada, ela terá poucos grupos únicos, 
        # e a mediana de cada grupo será um valor inteiro.

        # Calcula a média da coluna alvo por faixa (grupo) e Tipo IES
        df_grouped = combined_df.groupby([group_col, 'Tipo IES'])[target_column].mean().reset_index()
        
        # Plota o gráfico de linha de média
        sns.lineplot(
            data=df_grouped,
            x=group_col, # Eixo X usa os grupos (0 a n-1)
            y=target_column,
            hue='Tipo IES',
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'},
            # CORREÇÃO 1: Substitui 'ci=None' pelo parâmetro não depreciado 'errorbar=None'
            errorbar=None 
        )
        
        # Ajusta os rótulos do eixo X (mostra o valor mediano real da feature para cada grupo)
        median_labels = combined_df.groupby(group_col)[feature].median().tolist()
        
        # Formata os rótulos para exibição (apenas para exibição)
        if feature == 'pib':
            # Formato de R$ milhões para PIB
            formatted_labels = [f'R${x/1e6:,.0f}M' for x in median_labels]
        elif feature == 'duracao_ideal_anos':
            # NOVO CASO: Duração em anos (sem casas decimais)
            formatted_labels = [f'{x:,.0f} anos' for x in median_labels]
        elif feature in ['igc', 'inscritos_por_vaga']:
            # Formato com uma casa decimal para IGC e Inscritos por Vaga
            formatted_labels = [f'{x:,.1f}' for x in median_labels]
        else:
            # Formato genérico com 1 casa decimal
            formatted_labels = [f'{x:,.1f}' for x in median_labels]
        
        # CORREÇÃO 2: Garante que o número de ticks seja IGUAL ao número de labels gerados.
        num_actual_labels = len(formatted_labels)
        
        # Aplica os rótulos e rotaciona para legibilidade
        x_label = f'{x_axis_title} (Mediana por Grupo)'
        plt.gca().set_xticks(range(num_actual_labels)) # Define o número CORRETO de ticks
        plt.gca().set_xticklabels(formatted_labels, rotation=45, ha='right') 
        plt.xlabel(x_label, fontsize=12)

        plot_type = f'LINHA_MEDIA_{num_actual_labels}Q' # Tipo de plotagem
                
        # --- Configurações Finais ---
        plt.title(f'Relação de "{y_axis_title}" vs "{x_axis_title}" ({plot_type})', fontsize=16, fontweight='bold') 
        plt.ylabel(y_axis_title, fontsize=12)
        plt.legend(title='Tipo IES')
        plt.tight_layout()
        
        plot_path = os.path.join(FIGURES_PATH, f'analise_{plot_type}_{target_column}_vs_{feature.lower()}.png')
        plt.savefig(plot_path)
        plt.close()

        print(f"-> Gráfico de {plot_type} para '{feature}' salvo em: {plot_path}")
            
    print("\n--- Análise concluída com sucesso! ---")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera gráficos descritivos comparando IES Públicas e Privadas a partir de arquivos CSV.')
    parser.add_argument(
        '--target',
        type=str,
        default='taxa_integralizacao',
        dest='target_column',
        help='A coluna numérica do CSV a ser usada como métrica no eixo Y (ex: taxa_integralizacao, igc, inscritos_por_vaga).'
    )
    args = parser.parse_args()

    create_comparative_charts(args.target_column)