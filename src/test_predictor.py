import joblib
import pandas as pd
import os

def load_models():
    """Carrega todos os modelos necessários e os retorna em um dicionário."""
    model_paths = {
        'RandomForest_publica': os.path.join('models', 'RandomForest_publica.joblib'),
        'RegressaoLogistica_publica': os.path.join('models', 'RegressaoLogistica_publica.joblib'),
        'RandomForest_privada': os.path.join('models', 'RandomForest_privada.joblib'),
        'RegressaoLogistica_privada': os.path.join('models', 'RegressaoLogistica_privada.joblib')
    }
    
    loaded_models = {}
    for name, path in model_paths.items():
        if not os.path.exists(path):
            print(f"ERRO: Modelo '{path}' não encontrado. Execute 'src/main.py' primeiro.")
            return None
        loaded_models[name] = joblib.load(path)
        
    print("Todos os modelos foram carregados com sucesso.")
    return loaded_models

def get_all_course_names():
    """Carrega e retorna uma lista de todos os nomes de cursos únicos."""
    cursos_path = os.path.join('data', 'transformed_data', 'cursos_tratados.csv')
    try:
        df_cursos = pd.read_csv(cursos_path, sep=';', encoding='latin1')
        return df_cursos['NO_CINE_ROTULO'].dropna().str.upper().unique()
    except FileNotFoundError:
        print(f"ERRO: Arquivo de cursos '{cursos_path}' não encontrado.")
        return []

def run_prediction(pipeline, test_data):
    """Executa a predição e retorna o resultado formatado."""
    if pipeline is None:
        return "Modelo não carregado", 0.0

    prediction = pipeline.predict(test_data)
    probabilities = pipeline.predict_proba(test_data)
    
    resultado = "ALTA EVASÃO" if prediction[0] == 1 else "BAIXA EVASÃO"
    prob_alta = probabilities[0][1]
    
    return resultado, prob_alta

def main():
    """
    Script para testar múltiplos modelos, imprimir resultados e salvá-los em um CSV.
    """
    print("--- INICIANDO TESTE EM LOTE DE TODOS OS CURSOS ---")

    NUM_CURSOS_A_TESTAR = 15 
    CENARIOS_ALUNOS = [50, 150, 300] 

    models = load_models()
    if models is None: return

    course_names = get_all_course_names()
    if not course_names.any(): return
        
    if NUM_CURSOS_A_TESTAR:
        print(f"\nAVISO: O teste será limitado a uma amostra de {NUM_CURSOS_A_TESTAR} cursos.")
        course_names = pd.Series(course_names).sample(NUM_CURSOS_A_TESTAR).values

    all_results = []
    print("\n" + "="*100)
    print(f"{'CURSO':<35} | {'ALUNOS':<8} | {'TIPO IES':<12} | {'MODELO':<20} | {'PREVISÃO':<15}")
    print("-" * 100)

    for curso in course_names:
        for n_alunos in CENARIOS_ALUNOS:
            for tipo_ies in ['publica', 'privada']:
                for model_name in ['RandomForest', 'RegressaoLogistica']:
                    
                    full_model_name = f"{model_name}_{tipo_ies}"
                    pipeline = models[full_model_name]
                    
                    tp_cat_admin = 1 if tipo_ies == 'publica' else 4
                    
                    # --- CORREÇÃO APLICADA AQUI ---
                    # Garante que TODAS as colunas esperadas pelo modelo estejam presentes.
                    test_data = pd.DataFrame({
                        # Features principais do teste
                        'NO_CINE_ROTULO': [curso],
                        'QT_MAT': [n_alunos],
                        'TP_CATEGORIA_ADMINISTRATIVA': [tp_cat_admin],
                        
                        # Features placeholder necessárias pelo pré-processador
                        'QT_CONC': [0],
                        'QT_SIT_DESVINCULADO': [0],
                        'NO_IES': ['UNIVERSIDADE TESTE'],
                        'SG_IES': ['UT'],
                        'NO_REGIAO_IES': ['Sudeste'],
                        'SG_UF_IES': ['SP']
                    })

                    resultado, prob_alta = run_prediction(pipeline, test_data)
                    print(f"{curso.title():<35} | {n_alunos:<8} | {tipo_ies.title():<12} | {model_name:<20} | {resultado} ({prob_alta:.1%})")
                    
                    all_results.append({
                        'curso': curso.title(), 'alunos': n_alunos,
                        'tipo_ies': tipo_ies.title(), 'modelo': model_name,
                        'previsao': resultado, 'prob_alta_evasao': prob_alta
                    })
        print("-" * 100)

    results_df = pd.DataFrame(all_results)
    output_path = os.path.join('reports', 'test_results_comparison.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\nResultados completos salvos em: {output_path}")

    print("\n--- TESTE FINALIZADO ---")

if __name__ == "__main__":
    main()