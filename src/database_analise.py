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
        'nm_categoria',
        'sigla_uf_curso'
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
        if feature == 'nm_categoria':
            # Aumenta significativamente a largura para acomodar os muitos rótulos
            plt.figure(figsize=(24, 10)) 
        else:
            # Tamanho padrão para os outros gráficos
            plt.figure(figsize=(16, 10))

        # Cria o gráfico de barras usando Seaborn
        ax = sns.barplot(
            x=feature, 
            y=target_column, 
            hue='Tipo IES', 
            data=combined_impact, 
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'}
        )

        # Títulos e labels
        plt.title(f'Análise da Média de "{y_axis_title}" por "{feature}"', fontsize=18, fontweight='bold')
        plt.ylabel(f'Média de {y_axis_title}', fontsize=14)
        plt.xlabel(feature, fontsize=14)
        
        # Ajustes no eixo X para melhor legibilidade
        if feature in ['nm_categoria', 'sigla_uf_curso']:
            # Se for uma dessas features com muitos rótulos, diminui a fonte
            plt.xticks(rotation=45, ha='right', fontsize=8)
        else:
            # Mantém o padrão para os outros gráficos
            plt.xticks(rotation=45, ha='right')
        
        # Adiciona os valores numéricos no topo de cada barra
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}', 
                            ha='center', va='bottom', fontsize=12, color='black')

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
    
    print(f"\n--- Gerando Tabela Resumo para Features Categóricas (Alvo: {target_column}) ---")

    # 1. Definir os dados de importância (SHAP) extraídos da imagem
    # (Feature, Tipo IES): Valor
    # Nota: Usamos apenas a INTERSECÇÃO entre as features categóricas do script
    # ('features_to_analyze') e as features da imagem SHAP.
    
    # Mapeamento de nomes do script para nomes da imagem:
    # 'faixa_etaria' -> 'Faixa Etaria'
    # 'tp_cor_raca' -> 'Tp Cor Raca'
    # 'tp_escola_conclusao_ens_medio' -> 'Tp Escola Conclusao Ens Medio'
    # 'tp_sexo' -> 'Tp Sexo'
    # 'nm_categoria' -> 'Nm Categoria'
    
    shap_data = {
        # ('Feature_Script', 'Tipo IES_Script'): Valor_SHAP_Imagem
        ('faixa_etaria', 'PUBLICA'): 0.169,
        ('faixa_etaria', 'PRIVADA'): 0.176,
        ('tp_cor_raca', 'PUBLICA'): 0.048,
        ('tp_cor_raca', 'PRIVADA'): 0.069,
        ('tp_escola_conclusao_ens_medio', 'PUBLICA'): 0.045,
        ('tp_escola_conclusao_ens_medio', 'PRIVADA'): 0.066,
        ('tp_sexo', 'PUBLICA'): 0.017,
        ('tp_sexo', 'PRIVADA'): 0.023,
        ('nm_categoria', 'PUBLICA'): 0.024,
        ('nm_categoria', 'PRIVADA'): 0.014,
        ('sigla_uf_curso', 'PUBLICA'): 0.018, 
        ('sigla_uf_curso', 'PRIVADA'): 0.021 
    }
    
    # Converte para uma Série pandas com MultiIndex
    s_shap = pd.Series(shap_data, name='Importancia_SHAP_Agregada')
    s_shap.index.names = ['Feature', 'Tipo IES']

    # 2. Identificar as features categóricas que temos dados SHAP
    # Filtra a lista 'features_to_analyze' para conter apenas as que estão em 's_shap'
    categorical_features_with_shap = [
        f for f in features_to_analyze if f in s_shap.index.get_level_values('Feature')
    ]

    # 3. Loop para calcular médias e juntar com dados SHAP
    all_tables = []
    for feature in categorical_features_with_shap:
        if feature not in combined_df.columns:
            print(f"AVISO: Feature '{feature}' para tabela resumo não encontrada no DataFrame. Pulando.")
            continue
            
        # Calcula a média da variável-alvo por categoria
        # .dropna() remove categorias nulas (ex: NaN em 'tp_sexo' se houver)
        df_mean = combined_df.dropna(subset=[feature])\
                             .groupby(['Tipo IES', feature])[target_column]\
                             .mean().reset_index()
        
        # Renomeia colunas para o relatório final
        df_mean = df_mean.rename(columns={
            feature: 'Categoria', 
            target_column: f'Media_{target_column}'
        })
        
        # Adiciona o nome da feature principal
        df_mean['Feature'] = feature
        
        # Define o mesmo índice que a série SHAP para poder fazer o 'join'
        df_mean = df_mean.set_index(['Feature', 'Tipo IES'])
        
        # Junta a média por categoria com a importância agregada da feature
        combined_table_feature = df_mean.join(s_shap)
        
        all_tables.append(combined_table_feature)

    # 4. Combinar e exibir a tabela final
    if all_tables:
        # Concatena todas as tabelas de features (formato longo)
        final_report_table_long = pd.concat(all_tables).reset_index()
        
        print("\n--- Gerando Tabela Resumo Pivotada (Formato Largo) ---")

        # --- INÍCIO DA MODIFICAÇÃO (Pivotar a Tabela) ---
        
        # 1. Pivotar a tabela
        # Index: O que permanece como linha (Feature e sua Categoria)
        # Columns: O que vai virar coluna (PUBLICA / PRIVADA)
        # Values: Os valores que serão distribuídos
        try:
            pivoted_table = final_report_table_long.pivot_table(
                index=['Feature', 'Categoria'],
                columns=['Tipo IES'],
                values=[f'Media_{target_column}', 'Importancia_SHAP_Agregada']
            )

            # 2. Achatar os MultiIndex das colunas 
            # (ex: ('Media_taxa_integralizacao', 'PUBLICA') -> 'Media_taxa_integralizacao_PUBLICA')
            pivoted_table.columns = [f'{val_col}_{tipo_ies_col}' for val_col, tipo_ies_col in pivoted_table.columns.values]

            # 3. Resetar o índice para que 'Feature' e 'Categoria' voltem a ser colunas
            final_report_table = pivoted_table.reset_index()

            # 4. (Opcional) Reordenar as colunas para uma melhor visualização (agrupar por métrica)
            col_media_publica = f'Media_{target_column}_PUBLICA'
            col_media_privada = f'Media_{target_column}_PRIVADA'
            col_shap_publica = 'Importancia_SHAP_Agregada_PUBLICA'
            col_shap_privada = 'Importancia_SHAP_Agregada_PRIVADA'

            # Define a ordem desejada, garantindo que as colunas existam
            desired_order = ['Feature', 'Categoria']
            
            # Adiciona colunas de Média se existirem
            if col_media_publica in final_report_table.columns:
                desired_order.append(col_media_publica)
            if col_media_privada in final_report_table.columns:
                desired_order.append(col_media_privada)
                
            # Adiciona colunas de SHAP se existirem
            if col_shap_publica in final_report_table.columns:
                desired_order.append(col_shap_publica)
            if col_shap_privada in final_report_table.columns:
                desired_order.append(col_shap_privada)
                
            # Captura quaisquer outras colunas (embora não deva haver)
            other_cols = [c for c in final_report_table.columns if c not in desired_order]
            
            final_report_table = final_report_table[desired_order + other_cols]

        except Exception as e:
            print(f"ERRO ao pivotar a tabela: {e}")
            print("Continuando com a tabela longa original.")
            # Fallback para o comportamento antigo se o pivô falhar
            final_report_table = final_report_table_long.reindex(columns=[
                'Feature', 
                'Tipo IES', 
                'Categoria', 
                f'Media_{target_column}', 
                'Importancia_SHAP_Agregada'
            ])
            
        # --- FIM DA MODIFICAÇÃO ---
        
        # Ajusta opções do pandas para imprimir a tabela completa
        pd.set_option('display.max_rows', None)
        pd.set_option('display.max_columns', None)
        pd.set_option('display.width', 1000)
        
        print("\nTabela de Resumo (Importância Agregada vs. Média da Variável Alvo por Categoria):")
        # Usar to_string() para garantir que tudo seja impresso
        print(final_report_table.to_string(index=False))
        
        # Opcional: Salvar em CSV
        report_table_path = os.path.join(REPORTS_PATH, f'resumo_categorico_{target_column}.csv')
        final_report_table.to_csv(report_table_path, index=False, sep=';', decimal=',')
        print(f"\nTabela de resumo salva em: {report_table_path}")
        
    else:
        print("Nenhuma tabela de resumo foi gerada (nenhuma feature categórica em comum encontrada).")

        
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