import pandas as pd
import os
import seaborn as sns
import matplotlib.pyplot as plt
import textwrap

def shorten_text(text, max_len=45):
    """Resume um texto se ele for maior que o comprimento máximo."""
    if len(text) > max_len:
        # textwrap.shorten é mais inteligente que um simples fatiamento
        return textwrap.shorten(text, width=max_len, placeholder="...")
    return text

def analyze_results():
    """
    Lê o arquivo de resultados dos testes e gera uma análise sobre
    concordância dos modelos e principais pontos de atenção.
    """
    results_path = os.path.join('reports', 'test_results_comparison.csv')
    
    try:
        df = pd.read_csv(results_path)
    except FileNotFoundError:
        print(f"ERRO: Arquivo de resultados '{results_path}' não encontrado.")
        print("Por favor, execute o script 'src/test_predictor.py' primeiro para gerar os dados.")
        return

    print("--- INICIANDO ANÁLISE DOS RESULTADOS DE PREVISÃO ---")

    # 1. Análise de Concordância entre Modelos
    df_pivot = df.pivot_table(index=['curso', 'alunos', 'tipo_ies'], 
                              columns='modelo', 
                              values='previsao', 
                              aggfunc='first').reset_index()

    df_pivot['concordancia'] = (df_pivot['RandomForest'] == df_pivot['RegressaoLogistica'])
    
    taxa_concordancia = df_pivot['concordancia'].mean()
    print(f"\n[Análise 1: Concordância dos Modelos]")
    print(f"Os modelos RandomForest e Regressão Logística concordaram em {taxa_concordancia:.2%} das previsões.")

    discordancias = df_pivot[~df_pivot['concordancia']].copy()
    if not discordancias.empty:
        print("\n--- Principais Casos de Discordância (Top 10) ---")
        for index, row in discordancias.head(10).iterrows():
            curso_resumido = shorten_text(row['curso'])
            print(f"\nCurso : {curso_resumido}")
            print(f"  - Cenário: {row['alunos']} alunos em IES {row['tipo_ies']}")
            print(f"  - RandomForest previu      : {row['RandomForest']}")
            print(f"  - RegressaoLogistica previu: {row['RegressaoLogistica']}")
    else:
        print("Não houve discordâncias entre os modelos nos testes executados.")

    # 2. Cursos com maior risco de evasão (segundo o RandomForest)
    df_rf = df[(df['modelo'] == 'RandomForest') & (df['previsao'] == 'ALTA EVASÃO')]
    
    if not df_rf.empty:
        cursos_criticos = df_rf.groupby('curso')['previsao'].count().sort_values(ascending=False)
        print("\n\n[Análise 2: Cursos com Maior Risco de Evasão (RandomForest)]")
        print("Cursos mais frequentemente previstos com 'ALTA EVASÃO' em diferentes cenários:")
        print(cursos_criticos.head(10))
        
        # Prepara os dados para o gráfico
        top_10_cursos = cursos_criticos.head(10)
        # APLICA A FUNÇÃO DE RESUMO NOS NOMES DOS CURSOS PARA O GRÁFICO
        shortened_labels = [shorten_text(label) for label in top_10_cursos.index]

        plt.figure(figsize=(12, 8))
        sns.barplot(x=top_10_cursos.values, y=shortened_labels, hue=shortened_labels, palette='viridis', dodge=False, legend=False)
        plt.title('Top 10 Cursos com Maior Frequência de Previsão de "Alta Evasão" (RandomForest)')
        plt.xlabel('Número de Cenários com Previsão de Alta Evasão')
        plt.ylabel('Curso')
        plt.tight_layout()
        
        plot_path = os.path.join('reports', 'cursos_maior_risco.png')
        plt.savefig(plot_path)
        print(f"\nGráfico salvo em: {plot_path}")
        plt.close()

    print("\n--- ANÁLISE FINALIZADA ---")


if __name__ == "__main__":
    analyze_results()