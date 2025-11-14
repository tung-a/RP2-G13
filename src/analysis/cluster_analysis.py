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

def analyze_cluster_profiles(df_with_clusters: pd.DataFrame, dataset_name: str, reports_path: str):
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
    
    # Identifica colunas numéricas (excluindo 'cluster' que é categórica)
    numeric_cols = df_with_clusters.select_dtypes(include=['number']).columns.tolist()
    
    # Identifica colunas categóricas/objeto
    categorical_cols = df_with_clusters.select_dtypes(include=['object', 'category']).columns.tolist()
    
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

def analyze_relative_importance(df_with_clusters: pd.DataFrame, numeric_cols: list, dataset_name: str, reports_path: str):
    """
    Calcula a importância relativa (Z-Score) das médias dos clusters em relação à média global.
    Gera um mapa de calor (heatmap) que facilita a visualização das características distintivas.

    Interpretação do Z-Score:
    - Valor positivo alto (> 0.5): O cluster tem um valor muito ACIMA da média global para essa feature.
    - Valor negativo alto (< -0.5): O cluster tem um valor muito ABAIXO da média global.
    - Próximo de 0: O cluster está na média.

    Args:
        df_with_clusters (pd.DataFrame): O DataFrame original com a coluna 'cluster'.
        numeric_cols (list): Lista das colunas numéricas a analisar.
        dataset_name (str): Nome do dataset (ex: 'publica').
        reports_path (str): Caminho para salvar os gráficos.
    """
    print(f"\n--- Analisando Importância Relativa (Z-Score) - {dataset_name.upper()} ---")
    
    try:
        # --- CORREÇÃO: Converter explicitamente para float para evitar erros com Int64/booleans no Seaborn ---
        data_numeric = df_with_clusters[numeric_cols].astype(float)
        cluster_labels = df_with_clusters['cluster']

        # Médias e Desvios Padrão Globais
        global_mean = data_numeric.mean()
        global_std = data_numeric.std()
        
        # Médias por Cluster
        cluster_means = data_numeric.groupby(cluster_labels, observed=True).mean()
        
        # Cálculo do Z-Score: (Média Cluster - Média Global) / Desvio Padrão Global
        # Adicionamos um pequeno valor ao std para evitar divisão por zero
        relative_importance = (cluster_means - global_mean) / (global_std + 1e-9)
        
        # Preencher NaNs com 0 (caso desvio padrão seja 0)
        relative_importance = relative_importance.fillna(0)

        # Salvar CSV
        path = os.path.join(reports_path, f'cluster_relative_importance_{dataset_name}.csv')
        relative_importance.to_csv(path)
        logger.info(f"Importância relativa salva em: {path}")
        
        # Plotar Heatmap
        plt.figure(figsize=(14, 10))
        # Transpomos (.T) para que features fiquem no eixo Y e clusters no X
        sns.heatmap(relative_importance.T, cmap='RdBu_r', center=0, annot=True, fmt=".2f", linewidths=.5)
        plt.title(f'Perfil Relativo dos Clusters (Z-Score) - {dataset_name.upper()}', fontsize=16)
        plt.xlabel('Cluster', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(reports_path, f'cluster_heatmap_relative_{dataset_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Heatmap de importância relativa salvo em: {plot_path}")

    except Exception as e:
        logger.error(f"Erro na análise de importância relativa para '{dataset_name}': {e}", exc_info=True)

def analyze_cluster_drivers(df_processed, cluster_labels, feature_names, dataset_name: str, reports_path: str):
    """
    Utiliza um classificador (Random Forest) para identificar quais variáveis são
    mais importantes para a distinção dos clusters formados.
    Também gera regras de decisão simples (texto) usando uma Árvore de Decisão.

    Args:
        df_processed (numpy array): Os dados processados (X) usados no K-Means.
        cluster_labels (numpy array): Os labels (y) gerados pelo K-Means.
        feature_names (list): Lista com os nomes das features.
        dataset_name (str): Nome do dataset.
        reports_path (str): Caminho para salvar os outputs.
    """
    print(f"\n--- Analisando Variáveis Discriminantes (Drivers) dos Clusters ({dataset_name}) ---")
    
    try:
        # 1. Treinar Random Forest para obter a importância das features
        # Usamos um modelo robusto para capturar relações complexas
        clf = RandomForestClassifier(n_estimators=100, random_state=42, max_depth=10, n_jobs=-1)
        clf.fit(df_processed, cluster_labels)
        
        # Criar DataFrame de importância
        importances = pd.DataFrame({
            'feature': feature_names,
            'importance': clf.feature_importances_
        }).sort_values('importance', ascending=False)
        
        # Salvar CSV de importância
        csv_path = os.path.join(reports_path, f'cluster_drivers_importance_{dataset_name}.csv')
        importances.to_csv(csv_path, index=False)
        print(f"Tabela de importância dos drivers salva em: {csv_path}")
        
        # Plotar as Top 20 variáveis que definem os clusters
        top_n = 20
        plt.figure(figsize=(12, 10))
        # Correção para o aviso de depreciação do Seaborn: atribuir 'y' ao 'hue' e setar legend=False
        sns.barplot(data=importances.head(top_n), x='importance', y='feature', hue='feature', palette='viridis', legend=False)
        plt.title(f'Top {top_n} Variáveis Discriminantes dos Clusters - {dataset_name.upper()}', fontsize=16)
        plt.xlabel('Importância (Gini)', fontsize=12)
        plt.ylabel('Feature', fontsize=12)
        plt.tight_layout()
        
        plot_path = os.path.join(reports_path, f'cluster_drivers_plot_{dataset_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"Gráfico de drivers salvo em: {plot_path}")
        
        # 2. (Opcional) Gerar Regras Textuais com Árvore de Decisão Simples
        # Usamos uma árvore pequena (max_depth=3) para gerar regras humanamente legíveis
        tree = DecisionTreeClassifier(max_depth=3, random_state=42)
        tree.fit(df_processed, cluster_labels)
        
        rules = export_text(tree, feature_names=list(feature_names))
        
        txt_path = os.path.join(reports_path, f'cluster_rules_{dataset_name}.txt')
        with open(txt_path, 'w') as f:
            f.write(f"Regras de Decisão Simplificadas para Clusters - {dataset_name.upper()}\n")
            f.write("="*80 + "\n\n")
            f.write(rules)
        print(f"Regras de decisão (texto) salvas em: {txt_path}")

    except Exception as e:
        logger.error(f"Erro na análise de drivers para '{dataset_name}': {e}", exc_info=True)