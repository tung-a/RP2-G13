import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os
import textwrap

def shorten_text(text, max_len=40):
    """Resume um texto se ele for maior que o comprimento máximo."""
    if len(text) > max_len:
        return textwrap.shorten(text, width=max_len, placeholder="...")
    return text

def analyze_efficiency():
    """
    Calcula e analisa a "Taxa de Eficiência de Conclusão" como um proxy
    para a permanência dos alunos nos cursos.
    """
    try:
        df = pd.read_csv('data/transformed_data/dados_integrados_finais.csv', sep=';', encoding='latin1')
    except FileNotFoundError:
        print("ERRO: Arquivo 'dados_integrados_finais.csv' não encontrado.")
        print("Por favor, execute o 'main.py' primeiro para gerar o arquivo de dados integrado.")
        return

    print("\n--- INICIANDO ANÁLISE DE EFICIÊNCIA DE CONCLUSÃO (PERMANÊNCIA) POR CURSO ---")

    total_saidas = df['QT_CONC'] + df['QT_SIT_DESVINCULADO']
    df_com_saidas = df[total_saidas > 0].copy()

    if df_com_saidas.empty:
        print("AVISO: Nenhum dado de saída (formados ou evadidos) encontrado.")
        return

    total_saidas_filtrado = df_com_saidas['QT_CONC'] + df_com_saidas['QT_SIT_DESVINCULADO']
    df_com_saidas['EFICIENCIA_CONCLUSAO'] = (df_com_saidas['QT_CONC'] / total_saidas_filtrado)

    media_eficiencia = df_com_saidas.groupby('NO_CINE_ROTULO')['EFICIENCIA_CONCLUSAO'].mean().sort_values()

    # >>> CORREÇÃO APLICADA AQUI <<<
    # Para a análise dos "piores", vamos focar nos cursos que tiveram alguma eficiência (maior que 0),
    # mas ainda assim foram os mais baixos.
    piores_com_formandos = media_eficiencia[media_eficiencia > 0].head(10)
    
    # A análise dos melhores continua a mesma
    cursos_alta_eficiencia = media_eficiencia.tail(10).sort_values(ascending=False)

    print("\n[Análise: Cursos com Menor Eficiência (que tiveram formandos)]")
    print(piores_com_formandos)

    print("\n[Análise: Cursos com Maior Eficiência de Conclusão (Menor Evasão Relativa)]")
    print(cursos_alta_eficiencia)

    fig, axes = plt.subplots(1, 2, figsize=(18, 9))
    fig.suptitle('Análise de Eficiência de Conclusão por Curso', fontsize=20)

    # Gráfico para os piores cursos (que tiveram formandos)
    if not piores_com_formandos.empty:
        short_labels_baixa = [shorten_text(label) for label in piores_com_formandos.index]
        sns.barplot(ax=axes[0], x=piores_com_formandos.values, y=short_labels_baixa, palette='Reds_r', hue=short_labels_baixa, dodge=False, legend=False)
        axes[0].set_title('Top 10 Cursos com MENOR Eficiência (entre os que formam alunos)')
        axes[0].set_xlabel('Taxa de Eficiência de Conclusão Média')
        axes[0].set_ylabel('Curso')
    else:
        axes[0].text(0.5, 0.5, 'Não há dados suficientes para exibir\neste gráfico.', ha='center', va='center')

    # Gráfico para os melhores cursos
    if not cursos_alta_eficiencia.empty:
        short_labels_alta = [shorten_text(label) for label in cursos_alta_eficiencia.index]
        sns.barplot(ax=axes[1], x=cursos_alta_eficiencia.values, y=short_labels_alta, palette='Greens_r', hue=short_labels_alta, dodge=False, legend=False)
        axes[1].set_title('Top 10 Cursos com MAIOR Eficiência')
        axes[1].set_xlabel('Taxa de Eficiência de Conclusão Média')
        axes[1].set_ylabel('')
    else:
        axes[1].text(0.5, 0.5, 'Não há dados suficientes para exibir\neste gráfico.', ha='center', va='center')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join('reports', 'eficiencia_conclusao_por_curso.png')
    plt.savefig(plot_path)
    print(f"\nGráfico da análise de eficiência salvo em: {plot_path}")
    plt.close()

    print("\n--- ANÁLISE DE EFICIÊNCIA FINALIZADA ---")

if __name__ == "__main__":
    analyze_efficiency()