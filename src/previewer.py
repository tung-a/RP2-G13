import pandas as pd

# Caminhos para os arquivos de dados JÁ TRATADOS pelo csv_transformer.py
csv_cursos_path = 'data/transformed_data/cursos_tratados.csv'
csv_ies_path = 'data/transformed_data/ies_tratados.csv'
csv_enem_path = 'data/transformed_data/enem_tratados.csv'

def preview_dataframe(name, path):
    """
    Carrega um arquivo CSV e exibe um resumo informativo sobre ele.
    """
    try:
        print(f"--- Carregando preview para: {name} ---")
        df = pd.read_csv(path, sep=';', encoding='latin1', low_memory=False)

        print(f"\nFormato do DataFrame (Linhas, Colunas): {df.shape}")

        print("\nPrimeiras 5 linhas:")
        print(df.head())

        print("\nInformações sobre as colunas e tipos de dados:")
        df.info()

        print(f"\n--- Fim do preview para: {name} ---\n")

    except FileNotFoundError:
        print(f"!!! ERRO: Arquivo não encontrado em '{path}' !!!")
        print("Por favor, execute o script 'src/csv_transformer.py' primeiro para gerar os arquivos tratados.\n")
    except Exception as e:
        print(f"Ocorreu um erro inesperado ao processar o arquivo {path}: {e}\n")


if __name__ == "__main__":
    preview_dataframe("Cursos Tratados", csv_cursos_path)
    preview_dataframe("IES Tratados", csv_ies_path)
    preview_dataframe("ENEM Tratados", csv_enem_path)