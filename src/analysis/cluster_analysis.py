import logging
import pandas as pd
import os

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