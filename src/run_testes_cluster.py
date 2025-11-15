import os
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt
import joblib

# Importações de Modelagem e Pré-processamento
# ASSUMA QUE ESTAS FUNÇÕES ESTÃO DEFINIDAS NO SEU AMBIENTE
from kmodes.kprototypes import KPrototypes
import prince
from sklearn.preprocessing import StandardScaler

# ====================================================================
# VARIÁVEIS E FUNÇÕES DE MOCK (NECESSÁRIAS PARA ESTE BLOCO FUNCIONAR)
# VOCÊ DEVE SUBSTITUIR ISSO PELAS SUAS FUNÇÕES REAIS
# ====================================================================

# Lista de colunas categóricas (precisa ser definida globalmente)
COLUNAS_CATEGORICAS_CONHECIDAS = [
    'nu_ano_censo', 'tp_cor_raca', 'tp_sexo', 'in_financiamento_estudantil', 
    'in_apoio_social', 'tp_escola_conclusao_ens_medio', 'tp_grau_academico', 
    'tp_modalidade_ensino', 'tp_categoria_administrativa'
]

def preparar_dados_para_kprototypes_v2(df, colunas_categoricas_conhecidas):
    """ (Função de preparação robusta do K-Prototypes, adaptada para o uso) """
    df_prep = df.copy()

    for col in colunas_categoricas_conhecidas:
        if col in df_prep.columns:
            df_prep[col] = df_prep[col].astype('category')
            
    num_cols = df_prep.select_dtypes(include=np.number).columns
    cat_cols = df_prep.select_dtypes(include=['object', 'category']).columns
    
    # Imputação Simples (mediana/moda)
    for col in num_cols: df_prep[col] = df_prep[col].fillna(df_prep[col].median())
    for col in cat_cols: df_prep[col] = df_prep[col].fillna(df_prep[col].mode()[0])
    
    # Padronização
    if len(num_cols) > 0:
        scaler = StandardScaler()
        df_prep[num_cols] = scaler.fit_transform(df_prep[num_cols])
    
    cat_indices = [df_prep.columns.get_loc(c) for c in cat_cols]
    data_matrix = df_prep.values
    
    return data_matrix, cat_indices, cat_cols, df_prep # Retornando cat_cols para a visualização


def save_model(model, path):
    """ Função de mock para salvar o modelo """
    joblib.dump(model, path)
    return True

# FIM DAS FUNÇÕES DE MOCK
# ====================================================================


# Configurações de Caminho
MODELS_PATH = 'models'
REPORTS_PATH = 'reports/clustering'
os.makedirs(MODELS_PATH, exist_ok=True)
os.makedirs(REPORTS_PATH, exist_ok=True)

def run_clustering_tests(df: pd.DataFrame, name: str, k_min: int, k_max: int):
    """
    Executa a busca pelo melhor K (clusters) em um dataset, usando o Custo do K-Prototypes.
    """
    print(f"\n--- Iniciando testes de Clustering K-Prototypes para: {name.upper()} ---")

    # 1. Pré-processamento (Usando a função de K-Prototypes)
    try:
        # X_kproto é o array NumPy, cat_indices são os índices categóricos
        X_kproto, cat_indices, cat_col_names, X_kproto_df = preparar_dados_para_kprototypes_v2(df.copy(), COLUNAS_CATEGORICAS_CONHECIDAS)
    except Exception as e:
        print(f"ERRO DE PRÉ-PROCESSAMENTO: Falha ao preparar os dados. {e}")
        return

    cost_scores = []
    k_range = range(k_min, k_max + 1)
    
    # Não usaremos Silhouette Score, mas procuraremos o modelo com o menor Custo
    # (Embora o Cotovelo seja visual, guardar o melhor custo é uma boa prática)
    best_k_cost = -1
    best_cost_score = np.inf 
    best_kprototypes_model = None

    # 2. Busca pelo melhor K (usando o Custo)
    for k in k_range:
        try:
            # Treinamento do modelo K-Prototypes
            kprototypes_model = KPrototypes(
                n_clusters=k, 
                init='Cao', 
                n_init=10, # Aumentado n_init para maior robustez
                verbose=0,
                n_jobs=-1,
                random_state=42
            )
            
            # fit_predict retorna os labels, mas o fit calcula o cost_
            labels = kprototypes_model.fit_predict(X_kproto, categorical=cat_indices)
            
            # Coleta do Custo (Método do Cotovelo para K-Prototypes)
            cost = kprototypes_model.cost_
            cost_scores.append(cost)

            # Buscando o K que resultou no menor custo (Mínimo local é o ideal)
            if cost < best_cost_score:
                best_cost_score = cost
                best_k_cost = k
                best_kprototypes_model = kprototypes_model

            print(f"K={k}: Custo={cost:.2f} Melhor K até o momento: {best_k_cost})")
            
        except Exception as e:
            print(f"Erro ao testar K={k}: {e}. Pulando este K.")
            cost_scores.append(np.nan)
            continue
            
    # 3. Geração de Gráfico de Avaliação (Método do Cotovelo com Custo)

    # Gráfico do Método do Cotovelo (Custo)
    plt.figure(figsize=(10, 6))
    plt.plot(k_range, cost_scores, marker='o', color='blue')
    plt.title(f'Método do Cotovelo (Custo K-Prototypes) - {name.upper()}')
    plt.xlabel('Número de Clusters (K)')
    plt.ylabel('Custo (Dispersão Combinada)')
    plt.xticks(k_range)
    plt.grid(True)
    
    plot_path = os.path.join(REPORTS_PATH, f'figures/kprototypes_cost_metric_{name}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico do Método do Cotovelo salvo em: {plot_path}")

    # 4. Salvamento do Melhor Modelo
    if best_kprototypes_model:
        model_path = os.path.join(MODELS_PATH, f'KPrototypes_{name}_BEST_K{best_k_cost}.joblib')
        save_model(best_kprototypes_model, model_path)
        print(f"Melhor modelo K-Prototypes (K={best_k_cost}, Custo={best_cost_score:.4f}) salvo em: {model_path}")
        print(f"Gerando visualização do melhor cluster (K={best_k_cost}) usando TSNE...")

        try:
            # PCA não suporta dados mistos por isso usa-se o FAMD (Factor Analysis of Mixed Data)
            # a) Redução de Dimensionalidade com FAMD
            final_labels = best_kprototypes_model.predict(X_kproto, categorical=cat_indices)

            # b) Redução de Dimensionalidade (Abordagem Mista: FAMD + t-SNE)
            
            # 1. Primeiro, usamos o FAMD para converter dados mistos e reduzir o ruído
            # Vamos reduzir para 10 componentes (um bom ponto de partida)
            famd = prince.FAMD(n_components=10, random_state=42)
            # X_kproto_df é o DataFrame processado que sua função helper agora retorna
            X_famd_reduced = famd.fit_transform(X_kproto_df) 

            # 2. Agora, aplicamos o t-SNE sobre os 10 componentes do FAMD
            # O t-SNE é excelente para criar "ilhas" visuais
            from sklearn.manifold import TSNE
            tsne = TSNE(n_components=2, perplexity=30, max_iter=1000, random_state=42, n_jobs=-1)
            X_tsne = tsne.fit_transform(X_famd_reduced)

            # c) Plotagem (Usando os resultados do t-SNE)
            plt.figure(figsize=(8, 8))
            scatter = plt.scatter(
                X_tsne[:, 0], # Eixo X do t-SNE
                X_tsne[:, 1], # Eixo Y do t-SNE
                c=final_labels, 
                cmap='viridis', 
                s=20, 
                alpha=0.7
            )
            
            plt.title(f'Visualização K-Prototypes (K={best_k_cost}) - {name.upper()} - Redução t-SNE (via FAMD)')
            plt.xlabel('Componente t-SNE 1')
            plt.ylabel('Componente t-SNE 2')
            plt.legend(*scatter.legend_elements(), title=f"Clusters (K={best_k_cost})")
            
            viz_path = os.path.join(REPORTS_PATH, f'figures/cluster_viz_tsne_{name}_K{best_k_cost}_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')
            plt.savefig(viz_path)
            plt.close()
            print(f"Visualização de Cluster (t-SNE) salva em: {viz_path}")

        except Exception as e:
            print(f"AVISO: Falha ao gerar visualização FAMD. Erro: {e}")
            
    else:
        print("Nenhum modelo K-Prototypes válido pôde ser salvo.")

def main():
    # Carrega os dados de amostra (Ajuste o caminho se necessário)
    datasets = {
        'publica': pd.read_csv('data/publica_sample.csv'),
        'privada': pd.read_csv('data/privada_sample.csv')
    }

    parser = argparse.ArgumentParser(description='Busca pelo melhor número de clusters (K) para K-Means.')
    parser.add_argument('--k_min', type=int, default=2,
                        help='Valor mínimo de K (número de clusters) a ser testado. Deve ser >= 2.')
    parser.add_argument('--k_max', type=int, default=15,
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
