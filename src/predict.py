import joblib
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns

def predict_permanence(student_data, institution_type):
    """
    Carrega o modelo treinado e o pré-processador para prever o tempo de permanência
    de um novo aluno.
    """
    MODELS_PATH = 'models'
    model_path = os.path.join(MODELS_PATH, f'permanencia_model_{institution_type}.joblib')
    preprocessor_path = os.path.join(MODELS_PATH, f'preprocessor_{institution_type}.joblib')

    if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
        return f"Modelo para instituição '{institution_type}' não encontrado. Execute o main.py primeiro."

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    student_df = pd.DataFrame([student_data])
    processed_data = preprocessor.transform(student_df)
    prediction = model.predict(processed_data)

    return prediction[0]


def run_and_save_all_scenarios():
    """
    Gera, testa e guarda num CSV todas as combinações possíveis de perfis de alunos.
    """
    print("--- INICIANDO TESTE EXAUSTIVO DE TODOS OS CENÁRIOS ---")

    possible_values = {
        'tp_cor_raca': {0: "Não declarado", 1: "Branca", 2: "Preta", 3: "Parda", 4: "Amarela", 5: "Indígena"},
        'tp_sexo': {1: "Masculino", 2: "Feminino"},
        'tp_escola_conclusao_ens_medio': {1: "Privada", 2: "Pública"},
        'tp_modalidade_ensino': {1: "Presencial", 2: "EAD"},
        'in_financiamento_estudantil': {0: "Não", 1: "Sim"},
        'in_apoio_social': {0: "Não", 1: "Sim"}
    }
    
    base_values = {
        'faixa_etaria': 3,
        'tp_grau_academico': 1,
        'nu_carga_horaria': 3600
    }

    keys = possible_values.keys()
    value_codes = [list(v.keys()) for v in possible_values.values()]
    all_combinations = list(itertools.product(*value_codes))
    
    print(f"Total de cenários a serem calculados: {len(all_combinations) * 2}")

    results = []
    
    for combo in all_combinations:
        student_profile = base_values.copy()
        for i, key in enumerate(keys):
            student_profile[key] = combo[i]

        for inst_type, inst_code in [('publica', 1), ('privada', 5)]:
            profile_for_prediction = student_profile.copy()
            profile_for_prediction['tp_categoria_administrativa'] = inst_code
            
            prediction = predict_permanence(profile_for_prediction, inst_type)

            if isinstance(prediction, float):
                profile_description = {
                    "Cor/Raça": possible_values['tp_cor_raca'][profile_for_prediction['tp_cor_raca']],
                    "Sexo": possible_values['tp_sexo'][profile_for_prediction['tp_sexo']],
                    "Escola Média": possible_values['tp_escola_conclusao_ens_medio'][profile_for_prediction['tp_escola_conclusao_ens_medio']],
                    "Modalidade": possible_values['tp_modalidade_ensino'][profile_for_prediction['tp_modalidade_ensino']],
                    "Financiamento": possible_values['in_financiamento_estudantil'][profile_for_prediction['in_financiamento_estudantil']],
                    "Apoio Social": possible_values['in_apoio_social'][profile_for_prediction['in_apoio_social']]
                }
                
                results.append({
                    "Tipo IES": inst_type.upper(),
                    **profile_description,
                    "Previsão (anos)": round(prediction, 2)
                })

    results_df = pd.DataFrame(results)
    
    REPORTS_PATH = 'reports'
    os.makedirs(REPORTS_PATH, exist_ok=True)
    output_path = os.path.join(REPORTS_PATH, 'prediction_scenarios.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Todos os {len(results_df)} cenários foram guardados em: {output_path} ---")
    
    return output_path

def analyze_predictions(csv_path):
    """
    Lê o ficheiro CSV com as previsões, realiza análises e GERA GRÁFICOS para extrair insights.
    """
    if not os.path.exists(csv_path):
        print(f"Ficheiro de cenários não encontrado em {csv_path}")
        return

    print("\n\n--- INICIANDO ANÁLISE DOS CENÁRIOS GERADOS ---")
    df = pd.read_csv(csv_path)
    
    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)

    for inst_type in ['PUBLICA', 'PRIVADA']:
        print(f"\n--- ANÁLISE PARA INSTITUIÇÕES DO TIPO: {inst_type} ---")
        subset_df = df[df['Tipo IES'] == inst_type].copy()
        
        # 1. Análise de Impacto por Característica (Textual e Gráfica)
        print("\n[Análise 1: Impacto Médio de Cada Característica na Previsão]")
        features_to_analyze = ["Modalidade", "Escola Média", "Financiamento", "Apoio Social", "Cor/Raça"]
        
        for feature in features_to_analyze:
            impact_analysis = subset_df.groupby(feature)['Previsão (anos)'].mean().sort_values(ascending=False)
            
            # Análise textual
            print(f"\nImpacto da característica '{feature}':")
            print(impact_analysis.to_string())
            if len(impact_analysis) > 1:
                diff = impact_analysis.max() - impact_analysis.min()
                print(f"-> Diferença máxima de impacto: {diff:.2f} anos")

            # Geração de Gráfico
            plt.figure(figsize=(10, 6))
            sns.barplot(x=impact_analysis.index, y=impact_analysis.values, palette='viridis')
            plt.title(f'Impacto Médio da Característica "{feature}"\nem IES do tipo {inst_type}')
            plt.ylabel('Previsão Média de Permanência (anos)')
            plt.xlabel(feature)
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # Guardar o gráfico
            plot_path = os.path.join(FIGURES_PATH, f'impacto_{feature.lower().replace("/", "")}_{inst_type.lower()}.png')
            plt.savefig(plot_path)
            plt.close()
            print(f"-> Gráfico guardado em: {plot_path}")

        # 2. Análise dos Cenários Extremos
        print("\n\n[Análise 2: Perfis com Maiores e Menores Previsões]")
        if not subset_df.empty:
            max_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmax()]
            min_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmin()]
            
            print("\nCenário com MAIOR tempo de permanência previsto:")
            print(max_pred.to_string())
            
            print("\nCenário com MENOR tempo de permanência previsto:")
            print(min_pred.to_string())


if __name__ == '__main__':
    # Passo 1: Gerar e guardar todos os cenários num ficheiro CSV
    scenarios_csv_path = run_and_save_all_scenarios()
    
    # Passo 2: Analisar os resultados guardados e gerar gráficos
    if scenarios_csv_path:
        analyze_predictions(scenarios_csv_path)