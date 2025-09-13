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

def get_user_input():
    """Coleta dados de um curso hipotético a partir da entrada do usuário."""
    print("\n--- Entre com os dados do curso para prever a evasão ---")
    
    tipo_ies_input = input("O curso é de IES 'publica' ou 'privada'? ").lower()
    while tipo_ies_input not in ['publica', 'privada']:
        print("Entrada inválida. Por favor, digite 'publica' ou 'privada'.")
        tipo_ies_input = input("O curso é de IES 'publica' ou 'privada'? ").lower()
        
    no_cine_rotulo = input("Nome do curso (ex: CIÊNCIA DA COMPUTAÇÃO): ").upper()
    qt_mat = int(input("Quantidade de alunos matriculados: "))
    
    # Define o código da categoria administrativa com base na escolha
    # 1 = Federal (Pública), 4 = Privada com fins lucrativos
    tp_cat_admin = 1 if tipo_ies_input == 'publica' else 4
    
    # Criando o DataFrame para a predição
    data = {
        'NO_CINE_ROTULO': [no_cine_rotulo],
        'QT_MAT': [qt_mat],
        'QT_CONC': [0], # Valor placeholder, não usado como feature
        'QT_SIT_DESVINCULADO': [0], # Valor placeholder
        'NO_IES': ['UNIVERSIDADE EXEMPLO'],
        'SG_IES': ['UEX'],
        'TP_CATEGORIA_ADMINISTRATIVA': [tp_cat_admin],
        'NO_REGIAO_IES': ['Sudeste'],
        'SG_UF_IES': ['SP']
    }
    
    return pd.DataFrame(data), tipo_ies_input

def main():
    """
    Script principal para carregar um modelo e fazer uma predição
    com base nos dados fornecidos pelo usuário.
    """
    dados_curso, tipo_ies = get_user_input()
    
    # Escolhe o melhor modelo (RandomForest) para o tipo de IES
    model_name = f"RandomForest_{tipo_ies}.joblib"
    model_path = os.path.join('models', model_name)
    
    pipeline = load_model(model_path)
    
    if pipeline:
        print("\n--- Realizando Predição ---")
        
        # Realiza a predição e obtém as probabilidades
        prediction = pipeline.predict(dados_curso)
        probabilities = pipeline.predict_proba(dados_curso)
        
        # Interpreta o resultado
        resultado = "ALTA EVASÃO" if prediction[0] == 1 else "BAIXA EVASÃO"
        prob_baixa = probabilities[0][0]
        prob_alta = probabilities[0][1]
        
        print(f"\nResultado da Previsão: {resultado}")
        print(f"Confiança (Probabilidade de Baixa Evasão): {prob_baixa:.2%}")
        print(f"Confiança (Probabilidade de Alta Evasão):  {prob_alta:.2%}")

if __name__ == "__main__":
    main()