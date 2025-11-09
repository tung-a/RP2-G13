import joblib
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse 
import shap # Import do SHAP adicionado ao topo

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
    (ANÁLISE EXAUSTIVA)
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
    
    count = 0

    for combo in all_combinations:
        for igc_faixa, igc_value in igc_faixas.items():
            student_profile = base_values_template.copy()
            student_profile['igc'] = igc_value
            
            for i, key in enumerate(keys):
                student_profile[key] = combo[i]

            for inst_type, inst_code in [('publica', 1), ('privada', 5)]:
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
    output_path = os.path.join(REPORTS_PATH, f'prediction_scenarios_{model_name}.csv')
    results_df.to_csv(output_path, index=False)
    
    print(f"\n--- Todos os {len(results_df)} cenários foram guardados em: {output_path} ---")
    return output_path

def analyze_predictions(csv_path):
    """
    (ANÁLISE EXAUSTIVA - GRÁFICOS)
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

def analyze_model_with_shap(model_name):
    """
    (ANÁLISE SHAP)
    Carrega o modelo e o pré-processador e roda a análise SHAP
    para entender o impacto *real* de TODAS as features.
    """
    print(f"\n\n--- INICIANDO ANÁLISE DE INTERPRETABILIDADE (SHAP) PARA: {model_name} ---")
    
    MODELS_PATH = 'models'
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'shap')
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    for inst_type in ['publica', 'privada']:
        print(f"\nAnalisando modelo para IES: {inst_type.upper()}")
        
        # --- 1. Carregar Modelo e Preprocessor ---
        model_path = os.path.join(MODELS_PATH, f'{model_name}_{inst_type}_best.joblib')
        preprocessor_path = os.path.join(MODELS_PATH, f'preprocessor_{inst_type}.joblib')

        if not os.path.exists(model_path):
            model_path = os.path.join(MODELS_PATH, f'permanencia_model_{inst_type}.joblib')

        if not os.path.exists(model_path) or not os.path.exists(preprocessor_path):
            print(f"Modelo ou pré-processador para '{inst_type}' não encontrado. Pulando análise SHAP.")
            continue

        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)

        # --- 2. Carregar Dados de Fundo (ESSENCIAL) ---
        try:
            X_test = pd.read_csv(f'data/{inst_type}_sample.csv') 
        except FileNotFoundError:
            print(f"ERRO: Dados de teste (ex: 'data/{inst_type}_sample.csv') não encontrados.")
            print("A análise SHAP precisa de dados de exemplo para funcionar. Pulando...")
            continue
        except Exception as e:
            print(f"ERRO ao carregar 'data/{inst_type}_sample.csv': {e}. Pulando...")
            continue

        # --- 3. Pré-processar os Dados de Fundo ---
        X_test_cleaned = X_test.copy()
        
        TARGET_DTYPES = {
            'tp_cor_raca': 'object', 'tp_sexo': 'object', 'faixa_etaria': 'float64',
            'in_financiamento_estudantil': 'float64', 'in_apoio_social': 'float64',
            'tp_escola_conclusao_ens_medio': 'object', 'sigla_uf_curso': 'object',
            'tp_grau_academico': 'object', 'tp_modalidade_ensino': 'object',
            'nu_carga_horaria': 'int64', 'nm_categoria': 'object', 'pib': 'int64',
            'inscritos_por_vaga': 'float64', 'duracao_ideal_anos': 'float64',
            'tp_categoria_administrativa': 'object', 'no_regiao_ies': 'object',
            'igc': 'float64', 'taxa_integralizacao': 'float64'
        }
        
        X_test_cleaned = X_test_cleaned.loc[:, X_test_cleaned.columns.isin(TARGET_DTYPES.keys())]
        print("\nTratando NaN e coerção de tipos...")
        
        if 'tp_sexo' in X_test_cleaned.columns:
            X_test_cleaned['tp_sexo'] = X_test_cleaned['tp_sexo'].astype(object)
            X_test_cleaned.loc[X_test_cleaned['tp_sexo'] == 1, 'tp_sexo'] = True
            X_test_cleaned.loc[X_test_cleaned['tp_sexo'] == 2, 'tp_sexo'] = False
            X_test_cleaned['tp_sexo'] = X_test_cleaned['tp_sexo'].astype(bool)

        for col, dtype in TARGET_DTYPES.items():
            if col not in X_test_cleaned.columns or col == 'tp_sexo':
                continue 

            if dtype in ['float64', 'int64']:
                if dtype == 'int64':
                    X_test_cleaned[col] = pd.to_numeric(X_test_cleaned[col], errors='coerce').fillna(0).astype('int64')
                elif dtype == 'float64':
                    X_test_cleaned[col] = pd.to_numeric(X_test_cleaned[col], errors='coerce').fillna(0.0).astype('float64')
            elif dtype == 'object':
                X_test_cleaned[col] = X_test_cleaned[col].astype(str).fillna('0').astype('object')
        
        print("Tipos de dados finais (prontos para o preprocessor):")
        print(X_test_cleaned.info())
        
        print("\nIniciando preprocessor.transform()...")
        X_test_processed = preprocessor.transform(X_test_cleaned)
        print("preprocessor.transform() concluído.")
        
        if hasattr(X_test_processed, "toarray"):
             X_test_processed_dense = X_test_processed.toarray()
             print(f"Conversão de matriz esparsa para densa concluída. Shape: {X_test_processed_dense.shape}")
        else:
             X_test_processed_dense = X_test_processed

        try:
            feature_names = preprocessor.get_feature_names_out()
        except AttributeError:
            try:
                feature_names = preprocessor.named_steps['preprocessor'].get_feature_names_out()
            except Exception:
                print("Aviso: Não foi possível obter nomes de features do preprocessor. Usando nomes originais.")
                feature_names = X_test_cleaned.columns.tolist() 
                
        if len(feature_names) != X_test_processed_dense.shape[1]:
             print(f"Alerta: Discrepância de colunas! Nomes: {len(feature_names)}, Processadas: {X_test_processed_dense.shape[1]}")
             if len(X_test_cleaned.columns) == X_test_processed_dense.shape[1]:
                  feature_names = X_test_cleaned.columns.tolist()
             else:
                  feature_names = [f'feature_{i}' for i in range(X_test_processed_dense.shape[1])]

        X_test_processed_df = pd.DataFrame(X_test_processed_dense, columns=feature_names)

        # --- 4. Calcular e Plotar SHAP ---
        if model_name in ['RandomForest', 'LightGBM', 'GradientBoosting']:
            explainer = shap.TreeExplainer(model)
            print("Usando TreeExplainer (Rápido)...")
        else:
            print("Usando KernelExplainer (pode ser lento)...")
            def predict_fn(x):
                if isinstance(x, pd.DataFrame):
                    x = x.values
                if hasattr(x, "toarray"):
                    x = x.toarray()
                return model.predict(x)

            X_test_sample = shap.sample(X_test_processed_df, 100 if X_test_processed_df.shape[0] > 100 else X_test_processed_df.shape[0]) 
            explainer = shap.KernelExplainer(predict_fn, X_test_sample)

        print("Calculando valores SHAP...")
        shap_values = explainer.shap_values(X_test_processed_df) 
        print("Valores SHAP calculados.")

        # --- Gráfico 1: Summary Plot (Importância Global) ---
        plt.figure(figsize=(16, 10))
        shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
        plt.title(f'Importância Global das Features (SHAP) - {inst_type.upper()} ({model_name})')
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_PATH, f'shap_summary_bar_{inst_type}_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"-> Gráfico de importância SHAP (bar) salvo em: {plot_path}")

        # --- Gráfico 2: Beeswarm Plot (Impacto e Direção) ---
        plt.figure(figsize=(16, 10))
        shap.summary_plot(shap_values, X_test_processed_df, show=False)
        plt.title(f'Impacto Detalhado das Features (SHAP) - {inst_type.upper()} ({model_name})')
        plt.tight_layout()
        plot_path = os.path.join(FIGURES_PATH, f'shap_summary_beeswarm_{inst_type}_{model_name}.png')
        plt.savefig(plot_path, bbox_inches='tight')
        plt.close()
        print(f"-> Gráfico de impacto SHAP (beeswarm) salvo em: {plot_path}")

# =============================================================================
# BLOCO DE EXECUÇÃO PRINCIPAL (MAIN)
# =============================================================================
if __name__ == '__main__':
    
    # --- Configuração dos Argumentos ---
    parser = argparse.ArgumentParser(
        description='Executa análises de interpretabilidade (SHAP) ou simulação exaustiva de cenários para modelos de previsão de permanência.'
    )
    
    # Argumento 1: Escolha do Modelo
    parser.add_argument(
        '--model', 
        type=str, 
        default='RandomForest',
        choices=['RandomForest', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge'],
        help='Escolha o modelo base para executar a análise.'
    )
    
    # Argumento 2: Escolha do Tipo de Análise (NOVO AJUSTE)
    parser.add_argument(
        '--analysis', 
        type=str, 
        default='shap', # Pode mudar o default se preferir
        choices=['exhaustive', 'shap'],
        help='Escolha o tipo de análise: "exhaustive" (simula todos os cenários e gera gráficos) ou "shap" (análise de interpretabilidade).'
    )
    
    args = parser.parse_args()

    # --- Execução com base nos argumentos ---
    
    print(f"==================================================")
    print(f"Modelo selecionado: {args.model}")
    print(f"Tipo de análise selecionada: {args.analysis}")
    print(f"==================================================")

    if args.analysis == 'shap':
        analyze_model_with_shap(args.model)
    
    elif args.analysis == 'exhaustive':
        # Roda a simulação exaustiva e DEPOIS a análise dos resultados
        print("\nIniciando simulação de cenários (exaustiva)...")
        scenarios_csv_path = run_and_save_all_scenarios(args.model)
        
        if scenarios_csv_path and os.path.exists(scenarios_csv_path):
            # Se o CSV foi criado, analisa os resultados
            analyze_predictions(scenarios_csv_path)
        else:
            print("A análise exaustiva não produziu um ficheiro CSV. A análise dos resultados foi pulada.")
    
    print("\n--- FIM DA EXECUÇÃO ---")


# análise SHAP (SHapley Additive exPlanations) é uma técnica de interpretabilidade de modelos de Machine Learning que busca explicar a contribuição de cada variável (feature) para uma previsão específica.

# Em uma análise SHAP, as variáveis "Feature Value" e "SHAP Value" indicam o seguinte:

# 💡 Feature Value (Valor da Variável)
# O Feature Value é o valor real que uma variável específica assumiu para a instância (linha de dados, amostra) que está sendo analisada.

# Em outras palavras, é o dado de entrada daquela feature para fazer a previsão.

# Em gráficos SHAP, a cor do ponto costuma representar o Feature Value (por exemplo, vermelho para valores altos da feature e azul para valores baixos).

# 🎯 SHAP Value (Valor SHAP)
# O SHAP Value representa a contribuição dessa feature (com seu Feature Value específico) para a diferença entre a previsão do modelo para aquela instância e a previsão média (ou valor base) de todas as instâncias.

# Positivo SHAP Value: Indica que o valor da variável contribuiu para aumentar a previsão do modelo em relação ao valor base.

# Negativo SHAP Value: Indica que o valor da variável contribuiu para diminuir a previsão do modelo em relação ao valor base.

# O módulo (valor absoluto) do SHAP Value indica a magnitude da influência. Variáveis com altos valores absolutos são consideradas mais importantes para a previsão daquela instância.