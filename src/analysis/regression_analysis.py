import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os
import numpy as np

def analyze_feature_importance(model, preprocessor_pipeline, report_path):
    """
    Analisa e plota a importância das features do modelo.
    """
    try:
        # Extrair nomes das features do pipeline de forma robusta
        preprocessor = preprocessor_pipeline.named_steps['preprocessor']
        
        # Nomes das features numéricas
        num_features = preprocessor.named_transformers_['num'].feature_names_in_
        
        # Nomes das features categóricas
        cat_features_raw = preprocessor.named_transformers_['cat'].feature_names_in_
        cat_features_encoded = preprocessor.named_transformers_['cat'].get_feature_names_out(cat_features_raw)
        
        all_features = np.concatenate([num_features, cat_features_encoded])
    except Exception as e:
        print(f"Não foi possível extrair nomes das features. Erro: {e}. Pulando gráfico de importância.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20) # Aumentado para Top 20

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title('Top 20 Features Mais Importantes para Prever Tempo de Permanência')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    output_path = os.path.join(report_path, f'feature_importance_{report_path.split(os.sep)[-1]}.png') # Nome dinâmico para evitar sobrescrever
    plt.savefig(output_path)
    print(f"Gráfico de importância das features salvo em: {output_path}")
    plt.close()

def save_metrics_report(metrics_publica, metrics_privada, report_path):
    """
    Salva as métricas de avaliação dos modelos em um arquivo CSV.
    """
    metrics_df = pd.DataFrame({
        'instituicao': ['Pública', 'Privada'],
        'MSE': [metrics_publica['mse'], metrics_privada['mse']],
        'R2': [metrics_publica['r2'], metrics_privada['r2']]
    })
    
    output_path = os.path.join(report_path, 'model_performance_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Relatório de métricas salvo em: {output_path}")