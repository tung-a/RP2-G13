import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Importações de Modelagem e Pré-processamento (Assumindo que estão corretas)
from modeling.train import train_kmeans_model, predict_clusters, save_model
from preprocessing.preprocessor import preprocess_for_kmeans # Função que retorna o array NumPy

# Importações do Scikit-learn
from sklearn.metrics import silhouette_score
from sklearn.cluster import MiniBatchKMeans, KMeans
# from sklearn_extra.cluster import KMedoids
from sklearn.decomposition import PCA

# Configurações de Caminho
MODELS_PATH = 'models'
REPORTS_PATH = 'reports'
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

def run_clustering_tests(df: pd.DataFrame, name: str, k_min: int, k_max: int):
    """
    Executa a busca pelo melhor K (clusters) em um dataset, usando Inércia e Silhouette Score.
    """
    print(f"\n--- Iniciando testes de Clustering para: {name.upper()} ---")

    # 1. Pré-processamento
    try:
        # X_kmeans é o array NumPy pronto (numérico e escalado)
        X_kmeans, kmeans_feature_names = preprocess_for_kmeans(df.copy())
    except Exception as e:
        print(f"ERRO DE PRÉ-PROCESSAMENTO: Falha ao preparar os dados. {e}")
        return

    inertia_scores = []
    silhouette_scores = []
    k_range = range(k_min, k_max + 1)
    
    best_k_silhouette = -1
    best_silhouette_score = -1.0
    best_kmeans_model = None

    # 2. Busca pelo melhor K
    for k in k_range:
        try:
            # Treinamento do modelo (usando MiniBatchKMeans)
            kmeans_model = train_kmeans_model(X_kmeans, n_clusters=k)
            labels = predict_clusters(kmeans_model, X_kmeans)
            
            # Coleta da Inércia (Método do Cotovelo)
            inertia = kmeans_model.inertia_
            inertia_scores.append(inertia)

            # Coleta do Silhouette Score (Requer K > 1)
            if k > 1 and len(set(labels)) > 1:
                score = silhouette_score(X_kmeans, labels)
                silhouette_scores.append(score)

                if score > best_silhouette_score:
                    best_silhouette_score = score
                    best_k_silhouette = k
                    best_kmeans_model = kmeans_model
            else:
                silhouette_scores.append(np.nan)

            print(f"K={k}: Inércia={inertia:.2f}, Silhouette={score:.4f} (Melhor K atual: {best_k_silhouette})")
            
        except Exception as e:
            print(f"Erro ao testar K={k}: {e}. Pulando este K.")
            inertia_scores.append(np.nan)
            silhouette_scores.append(np.nan)
            continue
            
    # 3. Geração de Gráficos de Avaliação

    # Gráfico do Método do Cotovelo (Inércia)
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    plt.plot(k_range, inertia_scores, marker='o')
    plt.title(f'Método do Cotovelo (Inércia) - {name.upper()}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Inércia')

    # Gráfico do Silhouette Score
    plt.subplot(1, 2, 2)
    # Ajusta o K-range para Silhouette, que começa em K=2
    k_silhouette_range = [k for k in k_range if k > 1 and not np.isnan(silhouette_scores[k-k_min])]
    silhouette_plot_scores = [s for s in silhouette_scores if not np.isnan(s)]
    
    if silhouette_plot_scores:
        plt.plot(k_silhouette_range, silhouette_plot_scores, marker='o', color='red')
        plt.title(f'Silhouette Score - {name.upper()}')
        plt.xlabel('Número de Clusters (K)')
        plt.ylabel('Score de Silhueta Médio')
        
    plot_path = os.path.join(REPORTS_PATH, f'clustering_metrics_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráficos de avaliação salvos em: {plot_path}")

    # 4. Salvamento do Melhor Modelo
    if best_kmeans_model:
        model_path = os.path.join(MODELS_PATH, f'KMeans_{name}_BEST_K{best_k_silhouette}.joblib')
        save_model(best_kmeans_model, model_path)
        print(f"Melhor modelo K-Means (K={best_k_silhouette}, Silhouette={best_silhouette_score:.4f}) salvo em: {model_path}")
        print(f"Gerando visualização do melhor cluster (K={best_k_silhouette}) usando PCA...")

        try:
            # a) Redução de Dimensionalidade
            # Use PCA para reduzir os dados pré-processados para 2 componentes.
            pca = PCA(n_components=2)
            X_pca = pca.fit_transform(X_kmeans)
            
            # b) Atribuição de Labels
            # Prevemos os clusters usando o melhor modelo encontrado
            final_labels = predict_clusters(best_kmeans_model, X_kmeans)
            
            # c) Plotagem
            plt.figure(figsize=(8, 8))
            
            # Usamos as labels para colorir o gráfico
            scatter = plt.scatter(
                X_pca[:, 0], 
                X_pca[:, 1], 
                c=final_labels, 
                cmap='viridis', 
                s=20, 
                alpha=0.6
            )
            
            # d) Adicionar os Centróides ao Gráfico (também projetados em 2D)
            # Projetamos os centróides do melhor modelo com o mesmo PCA
            centroids_pca = pca.transform(best_kmeans_model.cluster_centers_)
            plt.scatter(
                centroids_pca[:, 0], 
                centroids_pca[:, 1], 
                marker='X', # Marcador diferente para o centróide
                s=200, 
                c='black', 
                label='Centróides'
            )
            
            plt.title(f'Visualização K-Means (K={best_k_silhouette}) - {name.upper()} - Redução PCA')
            plt.xlabel('Componente Principal 1')
            plt.ylabel('Componente Principal 2')
            plt.legend(*scatter.legend_elements(), title="Clusters")
            
            viz_path = os.path.join(REPORTS_PATH, f'cluster_viz_{name}_K{best_k_silhouette}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(viz_path)
            plt.close()
            print(f"Visualização de Cluster salva em: {viz_path}")

        except Exception as e:
            print(f"AVISO: Falha ao gerar visualização PCA. Verifique se há dados suficientes para PCA. Erro: {e}")
        
    else:
        print("Nenhum modelo K-Means válido pôde ser salvo.")

def main():
    # Carrega os dados de amostra (Ajuste o caminho se necessário)
    datasets = {
        'publica': pd.read_csv('data/publica_sample.csv'),
        'privada': pd.read_csv('data/privada_sample.csv')
    }

    parser = argparse.ArgumentParser(description='Busca pelo melhor número de clusters (K) para K-Means.')
    parser.add_argument('--k_min', type=int, default=2,
                        help='Valor mínimo de K (número de clusters) a ser testado. Deve ser >= 2.')
    parser.add_argument('--k_max', type=int, default=10,
                        help='Valor máximo de K (número de clusters) a ser testado.')
    
    args = parser.parse_args()

    if args.k_min < 2:
        print("Aviso: k_min foi ajustado para 2, pois K-Means requer pelo menos 2 clusters.")
        args.k_min = 2

    for name, df in datasets.items():
        if df.empty:
            print(f"\n--- DataFrame '{name}' está vazio. Pulando testes. ---")
            continue
        
        run_clustering_tests(df, name, args.k_min, args.k_max)

    print("\n--- Todos os testes de Clustering concluídos. ---")

if __name__ == '__main__':
    main()