import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def analyze_feature_importance(model, preprocessor_pipeline, report_path, model_name):
    """
    Analisa e plota a importância das features do modelo de forma robusta.
    """
    try:
        preprocessor = preprocessor_pipeline.named_steps['preprocessor']
        all_features = []

        # Itera sobre os transformadores que foram efetivamente usados no pipeline
        for name, transformer, columns in preprocessor.transformers_:
            if name == 'remainder' or not columns:
                continue
            
            if name == 'num':
                # As features numéricas mantêm os seus nomes originais
                all_features.extend(columns)
            elif name == 'cat':
                # Para o OneHotEncoder, obtemos os nomes das novas colunas geradas
                encoded_cat_names = transformer.get_feature_names_out(columns)
                all_features.extend(encoded_cat_names)
        
        if not all_features:
            print("Não foram encontradas features para analisar a importância.")
            return

    except Exception as e:
        print(f"Não foi possível extrair nomes das features. Erro: {e}. Pulando gráfico de importância.")
        return

    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
    feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)

    plt.figure(figsize=(12, 10))
    sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
    plt.title(f'Top 20 Features Mais Importantes ({model_name.upper()})')
    plt.xlabel('Importância')
    plt.ylabel('Feature')
    plt.tight_layout()
    
    # Gerar um nome de ficheiro único para cada gráfico
    output_path = os.path.join(report_path, f'feature_importance_{model_name}.png')
    plt.savefig(output_path)
    print(f"Gráfico de importância das features salvo em: {output_path}")
    plt.close()

def save_metrics_report(metrics_publica, metrics_privada, report_path):
    """
    Salva as métricas de avaliação dos modelos num ficheiro CSV.
    """
    metrics_df = pd.DataFrame({
        'instituicao': ['Pública', 'Privada'],
        'MSE': [metrics_publica['mse'], metrics_privada['mse']],
        'R2': [metrics_publica['r2'], metrics_privada['r2']]
    })
    
    output_path = os.path.join(report_path, 'model_performance_metrics.csv')
    metrics_df.to_csv(output_path, index=False)
    print(f"Relatório de métricas salvo em: {output_path}")