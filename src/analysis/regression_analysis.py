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

        importances = model.feature_importances_
        feature_importance_df = pd.DataFrame({'feature': all_features, 'importance': importances})
        
        feature_importance_df = feature_importance_df.sort_values(by='importance', ascending=False).head(20)

        # plt.figure(figsize=(12, 10))
        # sns.barplot(x='importance', y='feature', data=feature_importance_df, palette='viridis')
        # plt.title(f'Top 20 Features Mais Importantes ({model_name.upper()})')
        # plt.xlabel('Importância')
        # plt.ylabel('Feature')
        # plt.tight_layout()
        
        # # Gerar um nome de ficheiro único para cada gráfico
        # output_path = os.path.join(report_path, f'feature_importance_{model_name}.png')
        # plt.savefig(output_path)
        # print(f"Gráfico de importância das features salvo em: {output_path}")
        # plt.close()

        return feature_importance_df
    
    except Exception as e:
        print(f"Erro ao analisar a importância das features: {e}")  
    
def plot_combined_feature_importance(df_publica, df_privada, report_path):
    """
    Plota a importância das features para instituições públicas e privadas no mesmo gráfico.
    Remove as linhas de grade e formata os nomes das features.
    """
    try:
        if df_publica is None or df_privada is None:
            print("Dados de importância das features incompletos. Não é possível plotar o gráfico combinado.")
            return

        # Preparar os dados para plotagem
        df_publica['instituicao'] = 'Pública'
        df_privada['instituicao'] = 'Privada'
        combined_df = pd.concat([df_publica, df_privada])

        # --- MODIFICAÇÃO: Formatar nomes das features ---
        combined_df['feature'] = combined_df['feature'].str.replace('_', ' ').str.title()
        
        # Ordenar as features pela importância média para uma visualização mais limpa
        combined_df['feature'] = pd.Categorical(combined_df['feature'], categories=combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False).index, ordered=True)
        combined_df = combined_df.sort_values(by=['instituicao', 'feature'], ascending=[True, False])

        # --- MODIFICAÇÃO: Remover linhas de grade ---
        plt.style.use('seaborn-v0_8-white') # Um estilo mais limpo sem grades por padrão
        plt.figure(figsize=(14, 12))
        
        sns.barplot(
            x='importance', 
            y='feature', 
            hue='instituicao', 
            data=combined_df, 
            palette={'Pública': '#2c7fb8', 'Privada': '#41b6c4'}, 
            orient='h'
        )
        
        plt.title('Comparação da Importância das Features: Pública vs. Privada', fontsize=18, fontweight='bold')
        plt.xlabel('Importância da Feature', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.legend(title='Instituição', loc='lower right', fontsize=12)
        
        # Opcional: Remover as grades explicitamente se o estilo 'white' ainda as mostrar ou se quiser ter certeza.
        plt.gca().yaxis.grid(False) # Remove as grades do eixo Y
        plt.gca().xaxis.grid(False) # Remove as grades do eixo X

        # Adicionando rótulos de importância para cada barra
        for container in plt.gca().containers:
            for p in container.patches:
                width = p.get_width()
                plt.gca().text(
                    width + 0.001,
                    p.get_y() + p.get_height() / 2,
                    f'{width:.3f}', 
                    ha='left', 
                    va='center', 
                    fontsize=10
                )

        plt.tight_layout()
        output_path = os.path.join(report_path, 'feature_importance_combined.png')
        plt.savefig(output_path)
        print(f"Gráfico de importância das features combinado salvo em: {output_path}")
        plt.close()

    except Exception as e:
        print(f"Erro ao plotar gráfico combinado de importância das features: {e}")

def aggregate_feature_importance(df):
    """
    Agrupa features OHE (One-Hot Encoded) de volta para a feature original e soma a importância.
    
    Args:
        df (pd.DataFrame): DataFrame com colunas 'instituicao', 'feature', 'importance'.

    Returns:
        pd.DataFrame: DataFrame agregado com 'feature' original.
    """
    df_agg = df.copy()

    # Features conhecidas por serem numéricas ou binárias (não OHE) que não devem ser modificadas
    # Adicione ou remova conforme a saída do seu preprocessor.get_feature_names_out()
    simple_features_to_preserve = ['igc', 'faixa_etaria', 'nu_carga_horaria', 
                                   'pib', 'duracao_ideal_anos', 'inscritos_por_vaga', 
                                   'taxa_integralizacao', 'tp_sexo']
    
    def get_original_name(feature_name):
        # Se for uma feature simples, preserva o nome
        if feature_name in simple_features_to_preserve:
            return feature_name
        
        # Tenta identificar o separador do OneHotEncoder, que é frequentemente '__' ou '_'.
        # Usamos '_' como padrão e pegamos tudo antes da última ocorrência, a menos que 
        # seja uma feature simples.
        parts = feature_name.split('_')
        
        # Verifica se o nome tem múltiplas partes e não é um nome simples
        if len(parts) > 1 and parts[0] not in simple_features_to_preserve:
            # Junta todas as partes, exceto a última (que é o valor da categoria)
            original_name = '_'.join(parts[:-1])
            
            # Última verificação de segurança: se o nome original é vazio ou muito curto, retorna o nome completo
            return original_name if original_name else feature_name
        
        return feature_name

    # Aplica a função para criar a coluna de feature original
    df_agg['original_feature'] = df_agg['feature'].apply(get_original_name)

    # Agrupa e soma a importância
    df_final = df_agg.groupby(['instituicao', 'original_feature'])['importance'].sum().reset_index()
    df_final.columns = ['instituicao', 'feature', 'importance']
    
    return df_final

def plot_combined_feature_importance_agregada(df_publica_raw, df_privada_raw, report_path, model_name):
    """
    Agrupa features One-Hot Encoded e plota a importância agregada para 
    instituições públicas e privadas no mesmo gráfico.

    Args:
        df_publica_raw (pd.DataFrame): Importância das features para IES Pública (raw OHE features).
        df_privada_raw (pd.DataFrame): Importância das features para IES Privada (raw OHE features).
        report_path (str): Caminho para guardar o gráfico (ex: 'reports/figures').
        model_name (str): Nome do modelo para incluir no título do gráfico.
    """
    try:
        if df_publica_raw is None or df_privada_raw is None:
            print("Dados de importância das features incompletos. Não é possível plotar o gráfico agregado.")
            return

        # 1. Preparar e Agregar os Dados
        df_publica_raw['instituicao'] = 'Pública'
        df_privada_raw['instituicao'] = 'Privada'
        
        # Concatena os dados brutos antes da agregação para processamento em massa,
        # depois separa para agregação por instituição se necessário (mais robusto).
        
        # Agregação
        df_publica_agg = aggregate_feature_importance(df_publica_raw)
        df_privada_agg = aggregate_feature_importance(df_privada_raw)

        # Combina os DataFrames agregados
        combined_df = pd.concat([df_publica_agg, df_privada_agg])

        # 2. Formatar nomes das features (Limpeza)
        # Transforma nomes como 'tp_cor_raca' em 'Tp Cor Raca'
        combined_df['feature'] = combined_df['feature'].str.replace('_', ' ').str.title()
        
        # 3. Ordenar as features
        # Ordenar pela importância média para uma visualização mais limpa
        mean_importance = combined_df.groupby('feature')['importance'].mean().sort_values(ascending=False)
        combined_df['feature'] = pd.Categorical(combined_df['feature'], categories=mean_importance.index, ordered=True)
        combined_df = combined_df.sort_values(by=['instituicao', 'feature'], ascending=[True, False])

        # 4. Plotagem
        plt.style.use('seaborn-v0_8-white')
        plt.figure(figsize=(14, 10))
        
        sns.barplot(
            x='importance', 
            y='feature', 
            hue='instituicao', 
            data=combined_df, 
            palette={'Pública': '#2c7fb8', 'Privada': '#41b6c4'}, 
            orient='h'
        )
        
        # Títulos e Rótulos
        plt.title(f'Importância Agregada das Features: Pública vs. Privada ({model_name})', fontsize=18, fontweight='bold')
        plt.xlabel('Importância Agregada (Soma dos Valores SHAP Médios)', fontsize=14)
        plt.ylabel('Feature', fontsize=14)
        plt.legend(title='Instituição', loc='lower right', fontsize=12)
        
        # Remover grades
        plt.gca().yaxis.grid(False)
        plt.gca().xaxis.grid(False)

        # Adicionando rótulos de importância para cada barra
        for container in plt.gca().containers:
            for p in container.patches:
                width = p.get_width()
                plt.gca().text(
                    width + 0.001,
                    p.get_y() + p.get_height() / 2,
                    f'{width:.3f}', 
                    ha='left', 
                    va='center', 
                    fontsize=10
                )

        plt.tight_layout()
        os.makedirs(report_path, exist_ok=True)
        output_path = os.path.join(report_path, f'feature_importance_agregada_{model_name}.png')
        plt.savefig(output_path)
        plt.close()

        print(f"\nGráfico de importância das features AGREGADO salvo em: {output_path}")
        
    except Exception as e:
        print(f"Erro ao plotar gráfico combinado de importância das features agregadas: {e}")

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