import joblib
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse # Adicionado para argumentos de linha de comando

def predict_permanence(student_data, model_name, institution_type):
    """
    Carrega um modelo específico e o pré-processador para prever o tempo de permanência.
    """
    MODELS_PATH = 'models'
    # Alvo: Carregar o melhor modelo salvo pelo runtests.py
    model_path = os.path.join(MODELS_PATH, f'{model_name}_{institution_type}_best.joblib')
    preprocessor_path = os.path.join(MODELS_PATH, f'preprocessor_{institution_type}.joblib')

    # Fallback: Se o modelo do runtests não existir, tenta carregar o do main.py
    if not os.path.exists(model_path):
        fallback_path = os.path.join(MODELS_PATH, f'permanencia_model_{institution_type}.joblib')
        if os.path.exists(fallback_path):
            model_path = fallback_path
        else:
             return f"Modelo '{model_name}' para instituição '{institution_type}' não encontrado. Execute o runtests.py ou main.py primeiro."

    if not os.path.exists(preprocessor_path):
        return f"Pré-processador para '{institution_type}' não encontrado."

    model = joblib.load(model_path)
    preprocessor = joblib.load(preprocessor_path)

    student_df = pd.DataFrame([student_data])
    processed_data = preprocessor.transform(student_df)
    prediction = model.predict(processed_data)

    return prediction[0]

def run_and_save_all_scenarios(model_name):
    """
    Gera, testa, compara com o tempo ideal e guarda num CSV todas as combinações
    possíveis de perfis de alunos para um modelo específico.
    """
    print(f"--- INICIANDO TESTE EXAUSTIVO DE TODOS OS CENÁRIOS PARA O MODELO: {model_name} ---")

    possible_values = {
        'tp_cor_raca': {0: "Não declarado", 1: "Branca", 2: "Preta", 3: "Parda", 4: "Amarela", 5: "Indígena"},
        'tp_sexo': {1: "Masculino", 2: "Feminino"},
        'tp_escola_conclusao_ens_medio': {1: "Privada", 2: "Pública"},
        'tp_modalidade_ensino': {1: "Presencial", 2: "EAD"},
        'in_financiamento_estudantil': {0: "Não", 1: "Sim"},
        'in_apoio_social': {0: "Não", 1: "Sim"}
    }
    
    igc_faixas = {1: 0.5, 2: 1.5, 3: 2.5, 4: 3.5, 5: 4.5}
    base_values_template = {
        'faixa_etaria': 3,
        'tp_grau_academico': 1,
        'nu_carga_horaria': 3600,
        'duracao_ideal_anos': 4.0, # Assumindo 4 anos para um curso de 3600 horas
    }

    keys = possible_values.keys()
    value_codes = [list(v.keys()) for v in possible_values.values()]
    all_combinations = list(itertools.product(*value_codes))
    
    total_scenarios = len(all_combinations) * 2 * len(igc_faixas)
    print(f"Total de cenários a serem calculados: {total_scenarios}")
    results = []
    
    # NOVO: Contador para a barra de progresso
    count = 0

    for combo in all_combinations:
        for igc_faixa, igc_value in igc_faixas.items():
            student_profile = base_values_template.copy()
            student_profile['igc'] = igc_value
            
            for i, key in enumerate(keys):
                student_profile[key] = combo[i]

            for inst_type, inst_code in [('publica', 1), ('privada', 5)]:
                # NOVO: Atualiza e exibe o progresso
                count += 1
                if count % 1000 == 0 or count == total_scenarios:
                    print(f"  -> Calculando cenário {count} de {total_scenarios}...", end='\r')

                profile_for_prediction = student_profile.copy()
                profile_for_prediction['tp_categoria_administrativa'] = inst_code
                
                prediction = predict_permanence(profile_for_prediction, model_name, inst_type)

                if isinstance(prediction, float):
                    diferenca = prediction - profile_for_prediction['duracao_ideal_anos']
                    if diferenca < -0.5: status = 'Evasão Provável'
                    elif diferenca > 0.5: status = 'Atraso'
                    else: status = 'Conclusão no Prazo'
                    
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
                        "Previsão (anos)": round(prediction, 2),
                        "Duração Ideal (anos)": profile_for_prediction['duracao_ideal_anos'],
                        "Status Conclusão": status
                    })

    results_df = pd.DataFrame(results)
    
    REPORTS_PATH = 'reports'
    os.makedirs(REPORTS_PATH, exist_ok=True)
    # Salva o CSV com o nome do modelo para não sobrescrever os resultados
    output_path = os.path.join(REPORTS_PATH, f'prediction_scenarios_{model_name}.csv')
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

    print(f"\n\n--- INICIANDO ANÁLISE DOS CENÁRIOS GERADOS EM: {csv_path} ---")
    df = pd.read_csv(csv_path)
    
    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    model_name = os.path.basename(csv_path).replace('prediction_scenarios_', '').replace('.csv', '')

    # Análise 1: Distribuição do Status de Conclusão
    print("\n[Análise 1: Distribuição do Status de Conclusão Previsto]")
    plt.figure(figsize=(12, 7))
    sns.countplot(x='Status Conclusão', hue='Tipo IES', data=df, order=['Evasão Provável', 'Conclusão no Prazo', 'Atraso'], palette='mako')
    plt.title(f'Distribuição Prevista do Status de Conclusão ({model_name})', fontsize=16, fontweight='bold')
    plt.ylabel('Número de Cenários de Alunos', fontsize=12)
    plt.xlabel('Status de Conclusão Previsto', fontsize=12)
    status_plot_path = os.path.join(FIGURES_PATH, f'impacto_status_conclusao_{model_name}.png')
    plt.savefig(status_plot_path)
    plt.close()
    print(f"-> Gráfico de status de conclusão guardado em: {status_plot_path}")

    # Análise 2: Impacto Médio de Cada Característica
    print("\n[Análise 2: Impacto Médio de Cada Característica na Previsão]")
    features_to_analyze = ["Modalidade", "Escola Média", "Financiamento", "Apoio Social", "Cor/Raça", "Faixa IGC"]
    
    for feature in features_to_analyze:
        combined_impact = df.groupby(['Tipo IES', feature])['Previsão (anos)'].mean().reset_index()
        plt.style.use('seaborn-v0_8-whitegrid')
        plt.figure(figsize=(12, 8))
        ax = sns.barplot(x=feature, y='Previsão (anos)', hue='Tipo IES', data=combined_impact, palette={'PUBLICA': '#2c7fb8', 'PRIVADA': '#41b6c4'})
        plt.title(f'Impacto Médio da Característica "{feature}" ({model_name})', fontsize=16, fontweight='bold')
        plt.ylabel('Previsão Média de Permanência (anos)', fontsize=12)
        plt.xlabel(feature, fontsize=12)
        plt.xticks(rotation=45, ha='right')
        for container in ax.containers:
            for p in container.patches:
                height = p.get_height()
                ax.text(p.get_x() + p.get_width() / 2., height, f'{height:.2f}', ha='center', va='bottom', fontsize=10)
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_PATH, f'impacto_combinado_{feature.lower().replace("/", "")}_{model_name}.png')
        plt.savefig(plot_path)
        plt.close()
        print(f"-> Gráfico combinado para '{feature}' guardado em: {plot_path}")

    # Análise 3: Cenários Extremos
    print("\n\n[Análise 3: Perfis com Maiores e Menores Previsões]")
    for inst_type in ['PUBLICA', 'PRIVADA']:
        subset_df = df[df['Tipo IES'] == inst_type].copy()
        if not subset_df.empty:
            max_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmax()]
            min_pred = subset_df.loc[subset_df['Previsão (anos)'].idxmin()]
            
            print(f"\nCenário com MAIOR tempo de permanência previsto para IES {inst_type} (Modelo: {model_name}):")
            print(max_pred.to_string())
            
            print(f"\nCenário com MENOR tempo de permanência previsto para IES {inst_type} (Modelo: {model_name}):")
            print(min_pred.to_string())

if __name__ == '__main__':
    # NOVO: Adiciona argumentos para escolher o modelo
    parser = argparse.ArgumentParser(description='Gera e analisa cenários de previsão para um modelo específico.')
    parser.add_argument('--model', type=str, default='RandomForest',
                        choices=['RandomForest', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge'],
                        help='Escolha o modelo para usar nas previsões.')
    args = parser.parse_args()

    scenarios_csv_path = run_and_save_all_scenarios(args.model)
    
    if scenarios_csv_path:
        analyze_predictions(scenarios_csv_path)