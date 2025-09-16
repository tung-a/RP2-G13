import pandas as pd
import os

def get_csv_headers(file_path):
    """
    Lê a primeira linha de um arquivo CSV de forma eficiente para obter os nomes das colunas.
    Tenta detectar o separador (vírgula ou ponto e vírgula).
    """
    try:
        # Tenta com o separador vírgula, que é comum em seus novos arquivos
        header = pd.read_csv(file_path, sep=',', nrows=0).columns.tolist()
        return header
    except Exception:
        try:
            # Se falhar, tenta com ponto e vírgula e encoding latin1
            header = pd.read_csv(file_path, sep=';', encoding='latin1', nrows=0).columns.tolist()
            return header
        except Exception as e:
            return f"Não foi possível ler o arquivo. Erro: {e}"

def main():
    """
    Script principal para inspecionar os cabeçalhos dos arquivos de dados brutos.
    """
    print("--- INSPECIONANDO NOMES DAS COLUNAS NOS ARQUIVOS DE DADOS BRUTOS ---")

    # Caminhos para os novos arquivos que você quer inspecionar
    files_to_check = {
        "IES": os.path.join('data', 'ces', 'SoU_censo_IES', 'SoU_censo_IES.csv'),
        "Alunos 2009": os.path.join('data', 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_2009', 'SoU_censo_aluno_2009.csv'),
        "Alunos 2010": os.path.join('data', 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_2010', 'SoU_censo_aluno_2010.csv'),
        "Alunos 2011": os.path.join('data', 'ces', 'SoU_censo_alunos', 'SoU_censo_alunos_2011', 'SoU_censo_aluno_2011.csv'),
        "Cursos": os.path.join('data', 'ces', 'SoU_censo_cursos', 'SoU_censo_curso.csv'),
        "Docentes": os.path.join('data', 'ces', 'SoU_censo_docentes', 'SoU_censo_docente.csv'),
        "ENADE 2009": os.path.join('data', 'enade', 'SoU_enade', 'SoU_enade_2009', 'SoU_enade_2009.csv'),
        "ENADE 2010": os.path.join('data', 'enade', 'SoU_enade', 'SoU_enade_2010', 'SoU_enade_2010.csv'),
        "ENADE 2011": os.path.join('data', 'enade', 'SoU_enade', 'SoU_enade_2011', 'SoU_enade_2011.csv'),
    }

    for name, path in files_to_check.items():
        print(f"\n--- Colunas encontradas em: {name} ({path}) ---")
        
        if not os.path.exists(path):
            print("ARQUIVO NÃO ENCONTRADO.")
            continue
            
        columns = get_csv_headers(path)
        
        if isinstance(columns, list):
            for col in columns:
                print(f"- {col}")
        else:
            print(columns)  # Imprime a mensagem de erro, se houver

    print("\n--- INSPEÇÃO FINALIZADA ---")
    print("Use esta lista para verificar os nomes exatos das colunas necessárias para o seu pipeline.")

if __name__ == "__main__":
    main()