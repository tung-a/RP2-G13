import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os
import argparse
from scipy.stats import mannwhitneyu, ttest_ind

def perform_statistical_test(group1, group2, test_name='Mann-Whitney'):
    """
    Realiza um teste estatístico entre dois grupos e retorna o p-valor.
    """
    if len(group1) < 2 or len(group2) < 2:
        return np.nan

    if test_name == 'Mann-Whitney':
        stat, p_value = mannwhitneyu(group1, group2, alternative='two-sided')
    else:
        stat, p_value = ttest_ind(group1, group2, equal_var=False) # Welch's t-test
    
    return p_value

def create_comparative_charts(target_column):
    """
    Carrega os dados de IES públicas e privadas, gera gráficos comparativos
    e realiza validação estatística das diferenças.
    """
    # --- 1. Definição de Caminhos ---
    DATA_PATH = 'data'
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'descriptive_analysis')
    os.makedirs(FIGURES_PATH, exist_ok=True)

    print(f"--- INICIANDO ANÁLISE DESCRITIVA E ESTATÍSTICA (ALVO: {target_column}) ---")

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
    
    if target_column not in combined_df.columns:
        print(f"ERRO: A coluna-alvo '{target_column}' não existe no CSV. Verifique o nome.")
        return

    # --- 2.1 Validação Estatística Global ---
    print(f"\n[Estatística Global] Comparando {target_column} entre PUBLICA e PRIVADA...")
    pub_vals = df_publica[target_column].dropna()
    priv_vals = df_privada[target_column].dropna()
    
    p_val_global = perform_statistical_test(pub_vals, priv_vals)
    media_pub = pub_vals.mean()
    media_priv = priv_vals.mean()
    
    print(f"  -> Média Pública: {media_pub:.4f} | Média Privada: {media_priv:.4f}")
    print(f"  -> Teste Mann-Whitney U: p-valor = {p_val_global:.4e}")
    if p_val_global < 0.05:
        print("  -> RESULTADO: A diferença é ESTATISTICAMENTE SIGNIFICATIVA.")
    else:
        print("  -> RESULTADO: Não há evidência estatística de diferença.")

    # --- 3. Mapeamento de Códigos para Nomes Legíveis ---
    map_cor_raca = {
        0: 'Não Declarada', 1: 'Branca', 2: 'Preta', 3: 'Parda',
        4: 'Amarela', 5: 'Indígena', 9: 'Não Dispõe'
    }
    map_sexo = {1: 'Feminino', 2: 'Masculino', '1': 'Feminino', '2': 'Masculino'}
    map_escola = {1.0: 'Pública', 2.0: 'Privada', 9.0: 'Não Informado' }
    map_grau = {1.0: 'Bacharelado', 2.0: 'Licenciatura', 3.0: 'Tecnológico', 4.0: 'Bacharelado e Licenciatura', 5.0: 'Outros' }
    map_faixa_etaria = {
        1: '1 até 17', 2: '18 - 21', 3: '22 - 25', 4: '26 - 29',
        5: '30 - 33', 6: '34 - 37', 7: '38 - 41', 8: '42 - 45',
        9: '46 - 49', 10: '50 - 53', 11: '54 - 57', 12: '58 - 61',
        13: '62 - 65', 14: 'maior do que 65'
    }

    # Aplica o mapeamento
    # Nota: Convertemos para numérico primeiro para garantir que o map funcione se estiverem como string
    combined_df['tp_cor_raca'] = pd.to_numeric(combined_df['tp_cor_raca'], errors='coerce').map(map_cor_raca).fillna(combined_df['tp_cor_raca'])
    
    # Ajuste para tp_sexo que pode vir como boolean ou int
    if combined_df['tp_sexo'].dtype == bool:
         combined_df['tp_sexo'] = combined_df['tp_sexo'].map({False: 'Masculino', True: 'Feminino'}) # Assumindo ordem usual se for bool, mas ideal verificar origem
    else:
         combined_df['tp_sexo'] = combined_df['tp_sexo'].astype(str).replace({'1.0':'1', '2.0':'2'}).map(map_sexo).fillna(combined_df['tp_sexo'])

    combined_df['tp_escola_conclusao_ens_medio'] = pd.to_numeric(combined_df['tp_escola_conclusao_ens_medio'], errors='coerce').map(map_escola)
    combined_df['tp_grau_academico'] = pd.to_numeric(combined_df['tp_grau_academico'], errors='coerce').map(map_grau)
    combined_df['faixa_etaria'] = pd.to_numeric(combined_df['faixa_etaria'], errors='coerce').map(map_faixa_etaria)

    if 'nm_categoria' in combined_df.columns:
        combined_df['nm_categoria'] = combined_df['nm_categoria'].astype(str).str.replace(
        'Programas interdisciplinares abrangendo', 
        'Prog. Interdisciplinar', 
        regex=False
    )
    
    combined_df['duracao_ideal_anos'] = pd.to_numeric(combined_df['duracao_ideal_anos'], errors='coerce')
    if 'pib' in combined_df.columns:
        combined_df['pib_log'] = np.log1p(combined_df['pib'])

    # --- 4. Selecionar Features e Gerar Gráficos ---
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

    print(f"\nIniciando geração de {len(features_to_analyze)} gráficos e testes estatísticos por categoria...")
    y_axis_title = target_column.replace('_', ' ').title()

    stats_results = []

    for feature in features_to_analyze:
        if feature not in combined_df.columns:
            continue
        
        # Agrupa e calcula média
        combined_impact = combined_df.groupby(['Tipo IES', feature])[target_column].mean().reset_index()

        # --- Teste Estatístico por Categoria ---
        print(f"\n> Analisando significância para: {feature}")
        categories = combined_df[feature].dropna().unique()
        
        for cat in categories:
            subset = combined_df[combined_df[feature] == cat]
            pub_sub = subset[subset['Tipo IES'] == 'PUBLICA'][target_column].dropna()
            priv_sub = subset[subset['Tipo IES'] == 'PRIVADA'][target_column].dropna()
            
            p_val = perform_statistical_test(pub_sub, priv_sub)
            
            if pd.notna(p_val) and p_val < 0.05:
                diff_note = "SIGNIFICATIVO"
            else:
                diff_note = "Não sig."
            
            stats_results.append({
                'Feature': feature,
                'Categoria': cat,
                'P-Valor': p_val,
                'Significancia': diff_note,
                'N_Publica': len(pub_sub),
                'N_Privada': len(priv_sub)
            })
            # Opcional: imprimir apenas os significativos ou todos
            # print(f"  - {cat}: p={p_val:.4f} ({diff_note})")

        # --- Plotagem ---
        plt.style.use('seaborn-v0_8-whitegrid')
        if feature == 'nm_categoria':
            plt.figure(figsize=(24, 10)) 
        else:
            plt.figure(figsize=(16, 10))

        ax = sns.barplot(
            x=feature, 
            y=target_column, 
            hue='Tipo IES', 
            data=combined_impact, 
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'}
        )

        plt.title(f'Análise da Média de "{y_axis_title}" por "{feature}"', fontsize=18, fontweight='bold')
        plt.ylabel(f'Média de {y_axis_title}', fontsize=14)
        plt.xlabel(feature, fontsize=14)
        
        if feature in ['nm_categoria', 'sigla_uf_curso']:
            plt.xticks(rotation=45, ha='right', fontsize=8)
        else:
            plt.xticks(rotation=45, ha='right')
        
        # Adiciona valores nas barras
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                if pd.notna(height) and height > 0:
                    ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}', 
                            ha='center', va='bottom', fontsize=12, color='black')

        plt.tight_layout()
        plot_path = os.path.join(FIGURES_PATH, f'analise_{target_column}_por_{feature.lower().replace("/", "")}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"-> Gráfico salvo: {plot_path}")

    # Salvar relatório estatístico em CSV
    if stats_results:
        stats_df = pd.DataFrame(stats_results)
        stats_path = os.path.join(REPORTS_PATH, f'estatistica_comparativa_{target_column}.csv')
        stats_df.to_csv(stats_path, index=False, sep=';', decimal=',')
        print(f"\nRelatório de testes estatísticos salvo em: {stats_path}")

    # --- 5. Features Numéricas (Gráficos de Linha) ---
    numeric_features_to_analyze = ['igc', 'pib', 'inscritos_por_vaga', 'duracao_ideal_anos']
    numeric_features_to_analyze = [c for c in numeric_features_to_analyze if c in combined_df.columns]

    print(f"\nIniciando geração de {len(numeric_features_to_analyze)} gráficos numéricos...")

    for feature in numeric_features_to_analyze:
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        x_feature = feature
        x_axis_title = feature.replace('_', ' ').title() 
        num_quantiles = 10 
        group_col = f'{feature}_Group'
        
        try:
            combined_df[group_col] = pd.qcut(combined_df[x_feature], q=num_quantiles, labels=False, duplicates='drop')
        except ValueError:
            num_quantiles = 5
            try:
                combined_df[group_col] = pd.qcut(combined_df[x_feature], q=num_quantiles, labels=False, duplicates='drop')
            except ValueError:
                plt.close()
                continue
        
        df_grouped = combined_df.groupby([group_col, 'Tipo IES'])[target_column].mean().reset_index()
        
        sns.lineplot(
            data=df_grouped,
            x=group_col,
            y=target_column,
            hue='Tipo IES',
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'},
            errorbar=None 
        )
        
        median_labels = combined_df.groupby(group_col)[feature].median().tolist()
        
        if feature == 'pib':
            formatted_labels = [f'R${x/1e6:,.0f}M' for x in median_labels]
        elif feature == 'duracao_ideal_anos':
            formatted_labels = [f'{x:,.0f} anos' for x in median_labels]
        else:
            formatted_labels = [f'{x:,.1f}' for x in median_labels]
        
        num_actual_labels = len(formatted_labels)
        
        plt.gca().set_xticks(range(num_actual_labels))
        plt.gca().set_xticklabels(formatted_labels, rotation=45, ha='right') 
        plt.xlabel(f'{x_axis_title} (Mediana por Grupo)', fontsize=12)

        plot_type = f'LINHA_MEDIA_{num_actual_labels}Q'
        plt.title(f'Relação de "{y_axis_title}" vs "{x_axis_title}" ({plot_type})', fontsize=16, fontweight='bold') 
        plt.ylabel(y_axis_title, fontsize=12)
        plt.legend(title='Tipo IES')
        plt.tight_layout()
        
        plot_path = os.path.join(FIGURES_PATH, f'analise_{plot_type}_{target_column}_vs_{feature.lower()}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"-> Gráfico salvo: {plot_path}")
    
    print("\n--- Análise concluída com sucesso! ---")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gera gráficos descritivos e testes estatísticos comparando IES Públicas e Privadas.')
    parser.add_argument(
        '--target',
        type=str,
        default='taxa_integralizacao',
        dest='target_column',
        help='A coluna alvo (ex: taxa_integralizacao).'
    )
    args = parser.parse_args()

    create_comparative_charts(args.target_column)