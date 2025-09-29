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
    
    # Adicionando valores de IGC para cada faixa
    igc_faixas = {
        1: 0.5,
        2: 1.5,
        3: 2.5,
        4: 3.5,
        5: 4.5,
    }

    base_values_template = {
        'faixa_etaria': 3,
        'tp_grau_academico': 1,
        'nu_carga_horaria': 3600,
        'nu_ano_censo_y': 2019.0,
    }

    keys = possible_values.keys()
    value_codes = [list(v.keys()) for v in possible_values.values()]
    all_combinations = list(itertools.product(*value_codes))
    
    print(f"Total de cenários a serem calculados: {len(all_combinations) * 2 * len(igc_faixas)}")

    results = []
    
    for combo in all_combinations:
        for igc_faixa, igc_value in igc_faixas.items():
            student_profile = base_values_template.copy()
            student_profile['igc'] = igc_value
            student_profile['igc_faixa'] = igc_faixa
            
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
                        "Apoio Social": possible_values['in_apoio_social'][profile_for_prediction['in_apoio_social']],
                        "Faixa IGC": igc_faixa
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

    # 1. Análise de Impacto por Característica (Gráficos Combinados)
    print("\n[Análise 1: Impacto Médio de Cada Característica na Previsão (Gráficos Combinados)]")
    features_to_analyze = ["Modalidade", "Escola Média", "Financiamento", "Apoio Social", "Cor/Raça", "Faixa IGC"]
    
    for feature in features_to_analyze:
        # Agrupar os dados para calcular a média de previsão por característica e tipo de IES
        combined_impact = df.groupby(['Tipo IES', feature])['Previsão (anos)'].mean().reset_index()

        # Criar o gráfico combinado
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        
        ax = sns.barplot(
            x=feature,
            y='Previsão (anos)',
            hue='Tipo IES',
            data=combined_impact,
            palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'}
        )
        
        plt.title(f'Impacto Médio da Característica "{feature}" na Previsão de Permanência', fontsize=16, fontweight='bold')
        plt.ylabel('Previsão Média de Permanência (anos)', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        
        # Adicionar rótulos de dados
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                ax.text(
                    p.get_x() + p.get_width() / 2.,
                    height,
                    f'{height:.2f}',
                    ha='center',
                    va='bottom',
                    fontsize=10
                )
        
        plt.tight_layout()
        
        # Guardar o gráfico
        plot_path = os.path.join(FIGURES_PATH, f'impacto_combinado_{feature.lower().replace("/", "")}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"-> Gráfico combinado para '{feature}' guardado em: {plot_path}")

    # 2. Análise dos Cenários Extremos
    print("\n\n[Análise 2: Perfis com Maiores e Menores Previsões]")
    for inst_type in ['PUBLICA', 'PRIVADA']:
        subset_df = df[df['Tipo IES'] == inst_type].copy()
        if not subset_df.empty:
            max_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmax()]
            min_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmin()]
            
            print(f"\nCenário com MAIOR tempo de permanência previsto para IES {inst_type}:")
            print(max_pred.to_string())
            
            print(f"\nCenário com MENOR tempo de permanência previsto para IES {inst_type}:")
            print(min_pred.to_string())

if __name__ == '__main__':
    # Passo 1: Gerar e guardar todos os cenários num ficheiro CSV
    scenarios_csv_path = run_and_save_all_scenarios()
    
    # Passo 2: Analisar os resultados guardados e gerar gráficos
    if scenarios_csv_path:
        analyze_predictions(scenarios_csv_path)