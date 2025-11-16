import logging
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.ensemble import RandomForestClassifier

# Configura um logger para este módulo
logger = logging.getLogger(__name__)

def analyze_cluster_profiles(df_with_clusters: pd.DataFrame, categorical_cols: list, dataset_name: str, reports_path: str):
    """
    Analisa um DataFrame que contém uma coluna 'cluster' para entender o perfil de cada cluster.

    Calcula a distribuição de membros por cluster, a média das features numéricas
    e a moda (valor mais comum) das features categóricas para cada grupo.

    Salva os perfis em arquivos CSV no diretório de relatórios.

    Args:
        df_with_clusters (pd.DataFrame): O DataFrame completo com a coluna 'cluster' já adicionada.
        dataset_name (str): O nome do dataset (ex: 'publica', 'privada') para usar nos logs e nomes de arquivo.
        reports_path (str): O caminho para a pasta onde os relatórios CSV serão salvos.
    """
    
    logger.info(f"Iniciando análise de perfil dos clusters para '{dataset_name}'...")
    print(f"--- Etapa 3D: Analisando Perfil dos Clusters ({dataset_name}) ---")

    # 1. Validação
    if 'cluster' not in df_with_clusters.columns:
        logger.warning(f"Coluna 'cluster' não encontrada em '{dataset_name}'. Pulando análise de perfil.")
        return

    # 2. "Quais grupos existem?" - Distribuição dos Clusters
    try:
        cluster_distribution = df_with_clusters['cluster'].value_counts().sort_index()
        logger.info(f"Distribuição dos clusters para '{dataset_name}':\n{cluster_distribution.to_string()}")
        print(f"\nDistribuição dos Clusters ({dataset_name}):")
        print(cluster_distribution)
    except Exception as e:
        logger.error(f"Erro ao calcular distribuição de clusters para '{dataset_name}': {e}")
        return # Não podemos continuar sem a distribuição

    # 3. "Quais as diferenças?" - Perfis Numéricos e Categóricos
    
    numeric_cols = df_with_clusters.select_dtypes(include=['number']).columns.tolist()
    numeric_cols = [
        col 
        for col in numeric_cols 
        if col not in categorical_cols and col != 'cluster'
    ]
    
    # A coluna 'cluster' é categórica, mas não queremos analisá-la, queremos agrupar por ela.
    if 'cluster' in categorical_cols:
        categorical_cols.remove('cluster')
    
    # --- Perfil Numérico (Média) ---
    if numeric_cols:
        try:
            # Agrupa por cluster e calcula a média de todas as colunas numéricas
            numeric_profile = df_with_clusters.groupby('cluster', observed=True)[numeric_cols].mean()
            
            logger.info(f"Perfil Numérico (Médias) dos Clusters ({dataset_name}):\n{numeric_profile.to_string(max_cols=10, max_rows=10)}")
            print(f"\n--- Perfil Numérico (Médias) dos Clusters ({dataset_name}) ---")
            print(numeric_profile)
            
            # Salvar em CSV
            profile_path_num = os.path.join(reports_path, f'cluster_profile_numeric_{dataset_name}.csv')
            numeric_profile.to_csv(profile_path_num)
            logger.info(f"Perfil numérico salvo em: {profile_path_num}")

        except Exception as e:
            logger.error(f"Não foi possível gerar o perfil numérico para '{dataset_name}': {e}", exc_info=True)
    else:
        logger.info(f"Nenhuma coluna numérica encontrada para perfil de cluster em '{dataset_name}'.")

    # --- Perfil Categórico (Moda) ---
    if categorical_cols:
        try:
            # Agrupa por cluster e calcula a moda (valor mais frequente)
            # A lambda é necessária pois .mode() pode retornar múltiplos valores (pegamos o primeiro)
            categorical_profile = df_with_clusters.groupby('cluster', observed=True)[categorical_cols].agg(lambda x: x.mode().iloc[0] if not x.mode().empty else pd.NaT)

            logger.info(f"Perfil Categórico (Moda) dos Clusters ({dataset_name}):\n{categorical_profile.to_string(max_cols=10, max_rows=10)}")
            print(f"\n--- Perfil Categórico (Moda) dos Clusters ({dataset_name}) ---")
            print(categorical_profile)

            # Salvar em CSV
            profile_path_cat = os.path.join(reports_path, f'cluster_profile_categorical_{dataset_name}.csv')
            categorical_profile.to_csv(profile_path_cat)
            logger.info(f"Perfil categórico salvo em: {profile_path_cat}")

        except Exception as e:
            logger.error(f"Não foi possível gerar o perfil categórico para '{dataset_name}': {e}", exc_info=True)
    else:
        logger.info(f"Nenhuma coluna categórica encontrada para perfil de cluster em '{dataset_name}'.")
        
    print(f"--- Análise de Perfil ({dataset_name}) concluída. Relatórios salvos em '{reports_path}' ---")

def setup_clustering_paths(reports_path, dataset_name):
    cluster_reports_path = os.path.join(reports_path, 'cluster_reports', dataset_name)
    figures_path = os.path.join(reports_path, 'figures', dataset_name)
    os.makedirs(cluster_reports_path, exist_ok=True)
    os.makedirs(figures_path, exist_ok=True)
    return cluster_reports_path, figures_path

# --- FUNÇÃO AUXILIAR PARA AGREGAÇÃO ---
def _aggregate_categorical_zscores(df_zscore: pd.DataFrame, categorical_cols_original: list) -> pd.DataFrame:
    """
    Agrega Z-Scores de colunas One-Hot Encoded (OHE) de volta para a variável categórica original.
    O score agregado é a média do valor absoluto dos Z-Scores das categorias para cada cluster.
    """
    aggregated_scores = {}
    
    # Itera sobre a lista original de categóricas
    for original_col in categorical_cols_original:
        # Encontra todas as colunas que começam com o prefixo da variável original no DataFrame de Z-Scores
        prefix = f'{original_col}_'
        ohe_cols = [col for col in df_zscore.columns if col.startswith(prefix)]
        
        if ohe_cols:
            # Seleciona as colunas OHE
            df_ohe_zscores = df_zscore[ohe_cols]
            
            # 1. Calcula o Valor Absoluto para capturar o "poder de distinção" em ambas as direções
            df_abs_zscores = np.abs(df_ohe_zscores)
            
            # 2. Calcula a média ao longo das colunas (axis=1) -> Agrega por cluster
            aggregated_scores[original_col] = df_abs_zscores.mean(axis=1)
        else:
            logger.warning(f"Nenhuma coluna OHE encontrada para o prefixo '{prefix}'. Verifique a codificação.")

    # Converte o dicionário de scores agregados para DataFrame
    df_aggregated = pd.DataFrame(aggregated_scores)
    
    return df_aggregated

def analyze_relative_importance(df_with_clusters: pd.DataFrame, numeric_cols: list, dataset_name: str, reports_path: str):
    """
    Calcula a importância relativa (Z-Score) das médias dos clusters em relação à média global.
    Gera dois heatmaps e salva dois CSVs:
    1. Detalhado (Numéricas + Categóricas OHE).
    2. Agregado (Numéricas Originais + Categóricas Agregadas - Média do Absoluto).

    Args:
        df_with_clusters (pd.DataFrame): O DataFrame original com a coluna 'cluster'.
        numeric_cols (list): Lista das colunas numéricas a analisar.
        dataset_name (str): Nome do dataset (ex: 'publica').
        reports_path (str): Caminho para salvar os gráficos e relatórios.
    """
    cluster_reports_path, figures_path = setup_clustering_paths(reports_path, dataset_name)
    
    print(f"\n--- Analisando Importância Relativa (Z-Score) - {dataset_name.upper()} ---")
    
    try:
        cluster_labels = df_with_clusters['cluster']

        # 1. Identificar Colunas
        final_numeric_cols = [col for col in numeric_cols if col in df_with_clusters.columns and col != 'cluster']
        all_cols = set(df_with_clusters.columns)
        excluded_cols = set(final_numeric_cols) | {'cluster'}
        categorical_cols = [col for col in all_cols if col not in excluded_cols]
        
        logger.info(f"Colunas Numéricas (input): {final_numeric_cols}")
        logger.info(f"Colunas Categóricas (inferidas): {categorical_cols}")

        # 2. PRÉ-PROCESSAMENTO: Codificação e Combinação
        data_numeric_orig = df_with_clusters[final_numeric_cols].astype(float)
        
        if categorical_cols:
            df_categorical_encoded = pd.get_dummies(
                df_with_clusters[categorical_cols], 
                columns=categorical_cols, 
                prefix=categorical_cols, 
                drop_first=False
            ).astype(float)
            data_all = pd.concat([data_numeric_orig, df_categorical_encoded], axis=1)
        else:
            data_all = data_numeric_orig
            
        # 3. CÁLCULO DO Z-SCORE (em 'data_all')
        global_mean = data_all.mean()
        global_std = data_all.std()
        cluster_means = data_all.groupby(cluster_labels, observed=True).mean()
        
        # Cálculo do Z-Score para todas as features (numéricas e OHE)
        relative_importance = (cluster_means - global_mean) / (global_std + 1e-9)
        relative_importance = relative_importance.fillna(0) # Preenche NaNs com 0

        # 4. GERAÇÃO DO DATAFRAME AGREGADO (para a "visão mais específica")
        
        # A. Seleciona os Z-Scores das colunas numéricas originais
        relative_importance_numeric = relative_importance[final_numeric_cols]
        
        # B. Agrega os Z-Scores das colunas categóricas OHE
        if categorical_cols:
            df_aggregated_categorical = _aggregate_categorical_zscores(relative_importance.copy(), categorical_cols)
            
            # C. Combina os Z-Scores numéricos e os scores categóricos agregados
            relative_importance_aggregated = pd.concat([relative_importance_numeric, df_aggregated_categorical], axis=1)
            
            # --- SALVAR CSV AGREGADO ---
            path_agg = os.path.join(cluster_reports_path, f'cluster_relative_importance_aggregated_{dataset_name}.csv')
            relative_importance_aggregated.to_csv(path_agg)
            logger.info(f"Importância relativa AGREGADA (numéricas + categóricas agregadas) salva em: {path_agg}")
            
            # 5. PLOTAGEM DO HEATMAP AGREGADO
            
            # Plotar Heatmap Agregado (Visão de alto nível)
            plt.figure(figsize=(12, max(8, len(relative_importance_aggregated.columns) * 0.5)))
            
            sns.heatmap(
                relative_importance_aggregated.T, 
                cmap='viridis', 
                annot=True, 
                fmt=".2f", 
                linewidths=.5,
                cbar_kws={'label': 'Z-Score (Numéricas) / Média do |Z-Score| (Categóricas)'}
            )
            plt.title(f'Perfil Agregado dos Clusters (Z-Score e Média Absoluta) - {dataset_name.upper()}', fontsize=16)
            plt.xlabel('Cluster', fontsize=12)
            plt.ylabel('Feature (Numéricas Originais e Categóricas Agregadas)', fontsize=12)
            plt.tight_layout()
            
            plot_path_agg = os.path.join(figures_path, f'cluster_heatmap_relative_aggregated_{dataset_name}.png')
            plt.savefig(plot_path_agg)
            plt.close()
            print(f"Heatmap AGREGADO de importância relativa salvo em: {plot_path_agg}")
            
        else:
            print("Não há colunas categóricas para agregar. Gerando apenas o relatório numérico.")
            relative_importance_aggregated = relative_importance_numeric

        # --- GERAÇÃO E SALVAMENTO DO RELATÓRIO DETALHADO (OHE) ---
        
        # Salvar CSV Completo (Detalhado OHE)
        path = os.path.join(cluster_reports_path, f'cluster_relative_importance_full_{dataset_name}.csv')
        relative_importance.to_csv(path)
        logger.info(f"Importância relativa (numéricas + categóricas OHE) salva em: {path}")
        
        # Plotar Heatmap Completo (Detalhado OHE)
        plt.figure(figsize=(16, max(10, len(relative_importance.columns) * 0.3)))
        annot_val = True if len(relative_importance.columns) < 30 else False
        
        sns.heatmap(
            relative_importance.T, 
            cmap='RdBu_r', 
            center=0, 
            annot=annot_val, 
            fmt=".2f", 
            linewidths=.5,
            cbar_kws={'label': 'Z-Score'}
        )
        plt.title(f'Perfil Relativo dos Clusters (Z-Score) - {dataset_name.upper()} (Detalhado OHE)', fontsize=16)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Feature (Numéricas + Categóricas Codificadas OHE)', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(figures_path, f'cluster_heatmap_relative_detailed_{dataset_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Heatmap DETALHADO de importância relativa salvo em: {plot_path}")


    except Exception as e:
        logger.error(f"Erro na análise de importância relativa para '{dataset_name}': {e}", exc_info=True)
        print(f"Erro: {e}. Verifique se as colunas categóricas não contêm estruturas aninhadas (listas/arrays).")

def analyze_cluster_drivers(df_features: pd.DataFrame, cluster_labels: np.ndarray, categorical_cols: list, dataset_name: str, reports_path: str):
    """
    Utiliza um classificador (Random Forest) para identificar quais variáveis são
    mais importantes para a distinção dos clusters formados.
    
    Ajustado para aceitar o DataFrame de features e realizar o One-Hot Encoding
    nas colunas categóricas antes de treinar os modelos.

    Args:
        df_features (pd.DataFrame): O DataFrame de features (X) ANTES de ser transformado em array.
        cluster_labels (numpy array): Os labels (y) gerados pelo K-Means.
        categorical_cols (list): Lista das colunas categóricas (strings/objects) a serem codificadas.
        dataset_name (str): Nome do dataset.
        reports_path (str): Caminho para salvar os outputs.
    """
    cluster_reports_path, figures_path = setup_clustering_paths(reports_path, dataset_name)
    
    print(f"\n--- Analisando Variáveis Discriminantes (Drivers) dos Clusters ({dataset_name}) ---")
    
    try:
        # 1. Pré-processamento: Lidar com colunas categóricas (One-Hot Encoding)
        # Assumimos que as colunas numéricas restantes já estão limpas.
        df_encoded = pd.get_dummies(df_features, columns=categorical_cols, drop_first=False)
        
        # Garante que X é um array NumPy numérico para o Scikit-learn
        # Seleciona apenas as colunas que são numéricas no DataFrame codificado
        X = df_encoded.select_dtypes(include=np.number).values
        
        # Obtém os nomes finais das features após a codificação (IMPORTANTE!)
        feature_names_final = df_encoded.select_dtypes(include=np.number).columns.tolist()
        
        # Verifica consistência
        if len(X) != len(cluster_labels):
            logger.error("Erro: Número de observações no DataFrame de features não corresponde ao número de labels de cluster.")
            return

        # 2. Treinar Random Forest para obter a importância das features
        print("Treinando Random Forest para Feature Importance...")
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        
        # CORREÇÃO CRÍTICA: Usar X (array codificado) para treinar
        clf.fit(X, cluster_labels)
        
        # Criar DataFrame de importância
        importances = pd.DataFrame({
            # CORREÇÃO CRÍTICA: Usar feature_names_final
            'feature': feature_names_final,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Salvar CSV de importância
        csv_path = os.path.join(cluster_reports_path, f'cluster_drivers_importance_{dataset_name}.csv')
        importances.to_csv(csv_path, index=False)
        print(f"Tabela de importância dos drivers salva em: {csv_path}")
        
        # Plotar as Top 20 variáveis que definem os clusters
        top_n = 20
        plt.figure(figsize=(12, 10))
        sns.barplot(data=importances.head(top_n), x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title(f'Top {top_n} Variáveis Discriminantes dos Clusters - {dataset_name.upper()}', fontsize=16)
        plt.xlabel('Importância (Gini)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(figures_path, f'cluster_drivers_plot_{dataset_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico de drivers salvo em: {plot_path}")
        
        # 3. Gerar Regras Textuais com Árvore de Decisão Simples
        print("Gerando regras de decisão simples (Decision Tree)...")
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        
        # CORREÇÃO CRÍTICA: Usar X (array codificado) para treinar
        tree.fit(X, cluster_labels)
        
        # CORREÇÃO CRÍTICA: Usar feature_names_final para as regras
        rules = export_text(tree, feature_names=feature_names_final)
        
        txt_path = os.path.join(cluster_reports_path, f'cluster_rules_{dataset_name}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Regras de Decisão Simplificadas para Clusters - {dataset_name.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(rules)
        print(f"Regras de decisão (texto) salvas em: {txt_path}")

    except Exception as e:
        logger.error(f"Erro na análise de drivers para '{dataset_name}': {e}", exc_info=True)