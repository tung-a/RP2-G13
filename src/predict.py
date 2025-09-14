import joblib
import pandas as pd
import os

def load_model(model_path):
    """Carrega um modelo .joblib a partir do caminho especificado."""
    if not os.path.exists(model_path):
        print(f"ERRO: Arquivo do modelo não encontrado em '{model_path}'")
        print("Por favor, execute o pipeline principal ('src/main.py') para gerar os modelos primeiro.")
        return None
    
    print(f"Carregando modelo de: {model_path}")
    model = joblib.load(model_path)
    return model

def get_course_names():
    """Carrega e retorna uma lista de nomes de cursos únicos, já em maiúsculas."""
    cursos_path = os.path.join('data', 'transformed_data', 'cursos_tratados.csv')
    try:
        df_cursos = pd.read_csv(cursos_path, sep=';', encoding='latin1')
        # CORREÇÃO: Converte todos os nomes de curso para maiúsculas ao carregar
        return df_cursos['NO_CINE_ROTULO'].dropna().str.upper().unique()
    except FileNotFoundError:
        print(f"AVISO: Arquivo de cursos não encontrado em '{cursos_path}'. Não será possível listar os cursos.")
        return []

def get_user_input():
    """Coleta dados de um curso hipotético a partir da entrada do usuário."""
    print("\n--- Entre com os dados do curso para prever a evasão ---")
    
    tipo_ies_input = input("O curso é de IES 'publica' ou 'privada'? ").lower()
    while tipo_ies_input not in ['publica', 'privada']:
        print("Entrada inválida. Por favor, digite 'publica' ou 'privada'.")
        tipo_ies_input = input("O curso é de IES 'publica' ou 'privada'? ").lower()
    
    course_names = get_course_names()

    while True:
        prompt = "Nome do curso (digite 'LISTAR' para ver exemplos): "
        # CORREÇÃO: Adiciona .strip() para remover espaços e mantém .upper()
        no_cine_rotulo = input(prompt).strip().upper()
        
        if no_cine_rotulo == 'LISTAR':
            if len(course_names) > 0:
                print("\n--- Exemplo de Nomes de Cursos Válidos ---")
                sample_size = min(20, len(course_names))
                # Amostra da lista já em maiúsculas
                sample_list = pd.Series(course_names).sample(sample_size)
                # Mostra no formato "Title Case" para melhor leitura
                for name in sample_list:
                    print(f"- {name.title()}")
                print("-" * 40)
            else:
                print("Não foi possível carregar a lista de cursos.")
            continue
            
        if no_cine_rotulo not in course_names and len(course_names) > 0:
            print("AVISO: O nome do curso digitado não foi encontrado nos dados originais. A predição pode ser imprecisa.")
            
        break

    qt_mat_input = input("Quantidade de alunos matriculados: ")
    while not qt_mat_input.isdigit():
        print("Entrada inválida. Por favor, digite um número.")
        qt_mat_input = input("Quantidade de alunos matriculados: ")
    qt_mat = int(qt_mat_input)
    
    tp_cat_admin = 1 if tipo_ies_input == 'publica' else 4
    
    data = {
        'NO_CINE_ROTULO': [no_cine_rotulo],
        'QT_MAT': [qt_mat], 'QT_CONC': [0], 'QT_SIT_DESVINCULADO': [0],
        'NO_IES': ['UNIVERSIDADE EXEMPLO'], 'SG_IES': ['UEX'],
        'TP_CATEGORIA_ADMINISTRATIVA': [tp_cat_admin],
        'NO_REGIAO_IES': ['Sudeste'], 'SG_UF_IES': ['SP']
    }
    
    return pd.DataFrame(data), tipo_ies_input

def main():
    """
    Script principal para carregar um modelo e fazer uma predição
    com base nos dados fornecidos pelo usuário.
    """
    dados_curso, tipo_ies = get_user_input()
    
    model_name = f"RandomForest_{tipo_ies}.joblib"
    model_path = os.path.join('models', model_name)
    
    pipeline = load_model(model_path)
    
    if pipeline:
        print("\n--- Realizando Predição ---")
        
        prediction = pipeline.predict(dados_curso)
        probabilities = pipeline.predict_proba(dados_curso)
        
        resultado = "ALTA EVASÃO" if prediction[0] == 1 else "BAIXA EVASÃO"
        prob_baixa = probabilities[0][0]
        prob_alta = probabilities[0][1]
        
        print(f"\nResultado da Previsão: {resultado}")
        print(f"Confiança (Probabilidade de Baixa Evasão): {prob_baixa:.2%}")
        print(f"Confiança (Probabilidade de Alta Evasão):  {prob_alta:.2%}")

if __name__ == "__main__":
    main()