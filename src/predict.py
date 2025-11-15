import joblib
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse 
import shap 

# Importações do projeto para regenerar clusters
from preprocessing.preprocessor import preprocess_for_kmeans
from modeling.train import predict_clusters

# =============================================================================
# FUNÇÕES AUXILIARES DE CARGA E PREPARAÇÃO
# =============================================================================

def load_model_and_preprocessor(model_name, institution_type):
    """
    Carrega o modelo e o pré-processador salvos.
    PRIORIDADE: Carrega 'permanencia_model_*' (modelo final do main.py com clusters),
    e só depois tenta o '_best' (do runtests.py).
    """
    MODELS_PATH = 'models'
    
    # Caminhos
    path_main_model = os.path.join(MODELS_PATH, f'permanencia_model_{institution_type}.joblib')
    path_test_model = os.path.join(MODELS_PATH, f'{model_name}_{institution_type}_best.joblib')
    preprocessor_path = os.path.join(MODELS_PATH, f'preprocessor_{institution_type}.joblib')

    model_path = None
    
    # Lógica de Prioridade: Tenta primeiro o modelo completo (Main)
    if os.path.exists(path_main_model):
        model_path = path_main_model
        # print(f"  [Info] Usando modelo final do pipeline: {os.path.basename(model_path)}")
    elif os.path.exists(path_test_model):
        model_path = path_test_model
        print(f"  [Aviso] Modelo final não encontrado. Usando modelo de teste: {os.path.basename(model_path)}")
    else:
        print(f"AVISO: Nenhum modelo encontrado para '{institution_type}' (Nem final, nem teste).")
        return None, None

    if not os.path.exists(preprocessor_path):
        print(f"AVISO: Pré-processador não encontrado para '{institution_type}'.")
        return None, None

    try:
        model = joblib.load(model_path)
        preprocessor = joblib.load(preprocessor_path)
        return model, preprocessor
    except Exception as e:
        print(f"Erro ao carregar arquivos .joblib: {e}")
        return None, None

def get_kmeans_model(institution_type):
    """Carrega o modelo K-Means salvo."""
    path = f'models/kmeans_model_{institution_type}.joblib'
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_and_clean_data(institution_type):
    """
    Carrega, limpa os dados e ADICIONA A COLUNA DE CLUSTER necessária para o modelo.
    """
    data_path = f'data/{institution_type}_sample.csv'
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo de dados '{data_path}' não encontrado.")
        return None

    try:
        df = pd.read_csv(data_path)
    except Exception as e:
        print(f"ERRO ao ler CSV: {e}")
        return None

    # Definição de tipos para garantir consistência
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

    df_clean = df.copy()
    
    # Tratamento específico para tp_sexo
    if 'tp_sexo' in df_clean.columns:
        df_clean['tp_sexo'] = df_clean['tp_sexo'].astype(object)
        df_clean.loc[df_clean['tp_sexo'].isin([1, '1', '1.0']), 'tp_sexo'] = True
        df_clean.loc[df_clean['tp_sexo'].isin([2, '2', '2.0']), 'tp_sexo'] = False
        df_clean['tp_sexo'] = df_clean['tp_sexo'].astype(bool)

    # Coerção de tipos
    for col, dtype in TARGET_DTYPES.items():
        if col in df_clean.columns and col != 'tp_sexo':
            if dtype == 'int64':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0).astype('int64')
            elif dtype == 'float64':
                df_clean[col] = pd.to_numeric(df_clean[col], errors='coerce').fillna(0.0).astype('float64')
            elif dtype == 'object':
                df_clean[col] = df_clean[col].astype(str).fillna('0').astype('object')

    # --- Regenerar Clusters ---
    kmeans = get_kmeans_model(institution_type)
    if kmeans:
        print(f"  [Info] Aplicando K-Means ({institution_type}) para gerar coluna 'cluster'...")
        try:
            X_kmeans, _ = preprocess_for_kmeans(df_clean.copy())
            labels = predict_clusters(kmeans, X_kmeans)
            df_clean['cluster'] = labels
            df_clean['cluster'] = df_clean['cluster'].astype('category')
        except Exception as e:
            print(f"  [Aviso] Falha ao gerar clusters: {e}. O modelo pode falhar se 'cluster' for obrigatório.")
    else:
        print(f"  [Aviso] Modelo K-Means não encontrado para '{institution_type}'.")

    return df_clean

# =============================================================================
# FUNÇÕES DE PREVISÃO E CENÁRIOS
# =============================================================================

def predict_permanence(student_data, model_name, institution_type):
    """Prevê a permanência para um único perfil (dicionário)."""
    model, preprocessor = load_model_and_preprocessor(model_name, institution_type)
    if model is None: return None

    student_df = pd.DataFrame([student_data])
    
    # Tentativa de tratar cluster em predição single-shot
    if 'cluster' not in student_df.columns:
        try:
            feats = preprocessor.get_feature_names_out()
            if any('cluster' in f for f in feats):
                # Atribuição dummy (cluster 0) para não quebrar
                student_df['cluster'] = 0 
                student_df['cluster'] = student_df['cluster'].astype('category')
        except:
            pass

    processed_data = preprocessor.transform(student_df)
    prediction = model.predict(processed_data)
    return prediction[0]

def run_and_save_all_scenarios(model_name):
    """(ANÁLISE EXAUSTIVA) Gera todos os cenários possíveis e salva em CSV."""
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
        'duracao_ideal_anos': 4.0,
    }

    keys = possible_values.keys()
    value_codes = [list(v.keys()) for v in possible_values.values()]
    all_combinations = list(itertools.product(*value_codes))
    
    models_cache = {}
    kmeans_cache = {}
    total_scenarios = 0
    
    for inst_type in ['publica', 'privada']:
        models_cache[inst_type] = load_model_and_preprocessor(model_name, inst_type)
        kmeans_cache[inst_type] = get_kmeans_model(inst_type)
        n_clusters = kmeans_cache[inst_type].n_clusters if kmeans_cache[inst_type] else 1
        total_scenarios += len(all_combinations) * len(igc_faixas) * n_clusters

    print(f"Total de cenários a serem calculados: {total_scenarios}")
    results = []
    count = 0

    for combo in all_combinations:
        for igc_faixa, igc_value in igc_faixas.items():
            student_profile_base = base_values_template.copy()
            student_profile_base['igc'] = igc_value
            
            for i, key in enumerate(keys):
                student_profile_base[key] = combo[i]

            for inst_type, inst_code in [('publica', 1), ('privada', 5)]:
                model, preprocessor = models_cache[inst_type]
                if model is None: continue
                
                kmeans = kmeans_cache[inst_type]
                clusters_to_test = range(kmeans.n_clusters) if kmeans else [0]

                for cluster_id in clusters_to_test:
                    count += 1
                    if count % 1000 == 0: print(f"  -> Calculando cenário {count}/{total_scenarios}...", end='\r')

                    profile_for_prediction = student_profile_base.copy()
                    profile_for_prediction['tp_categoria_administrativa'] = inst_code
                    
                    if kmeans:
                        profile_for_prediction['cluster'] = cluster_id
                    
                    df_temp = pd.DataFrame([profile_for_prediction])
                    df_temp['tp_sexo'] = df_temp['tp_sexo'].map({1: False, 2: True}) 
                    if kmeans:
                        df_temp['cluster'] = df_temp['cluster'].astype('category')

                    try:
                        processed = preprocessor.transform(df_temp)
                        pred = model.predict(processed)[0]
                    except Exception:
                        continue

                    if isinstance(pred, float):
                        diferenca = pred - profile_for_prediction['duracao_ideal_anos']
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
                            "Faixa IGC": igc_faixa,
                            "Cluster": cluster_id if kmeans else "N/A"
                        }
                        
                        results.append({
                            "Tipo IES": inst_type.upper(),
                            **profile_description,
                            "Previsão (anos)": round(pred, 2),
                            "Duração Ideal (anos)": profile_for_prediction['duracao_ideal_anos'],
                            "Status Conclusão": status
                        })

    results_df = pd.DataFrame(results)
    REPORTS_PATH = 'reports'
    os.makedirs(REPORTS_PATH, exist_ok=True)
    output_path = os.path.join(REPORTS_PATH, f'prediction_scenarios_{model_name}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n--- Cenários salvos em: {output_path} ---")
    return output_path

def analyze_predictions(csv_path):
    """Gera gráficos a partir do CSV de cenários."""
    if not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    model_name = os.path.basename(csv_path).replace('prediction_scenarios_', '').replace('.csv', '')

    plt.figure(figsize=(12, 7))
    sns.countplot(x='Status Conclusão', hue='Tipo IES', data=df, order=['Evasão Provável', 'Conclusão no Prazo', 'Atraso'], palette='mako')
    plt.title(f'Distribuição Prevista do Status de Conclusão ({model_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURES_PATH, f'impacto_status_conclusao_{model_name}.png'))
    plt.close()

    features = ["Modalidade", "Escola Média", "Financiamento", "Apoio Social", "Cor/Raça", "Faixa IGC"]
    for feat in features:
        plt.figure(figsize=(12, 8))
        sns.barplot(x=feat, y='Previsão (anos)', hue='Tipo IES', data=df, palette='viridis')
        plt.title(f'Impacto: {feat} ({model_name})')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'impacto_combinado_{feat.lower().replace("/","")}_{model_name}.png'))
        plt.close()

# =============================================================================
# SIMULAÇÃO DE POLÍTICAS PÚBLICAS
# =============================================================================

def simulate_policy_interventions(model_name):
    """
    Simula o impacto de intervenções na taxa de integralização.
    """
    print(f"\n--- INICIANDO SIMULAÇÃO DE POLÍTICAS PÚBLICAS (Model: {model_name}) ---")
    
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'policy_simulation')
    os.makedirs(FIGURES_PATH, exist_ok=True)

    results_summary = []

    for inst_type in ['publica', 'privada']:
        print(f"\n> Simulando para: {inst_type.upper()}")
        
        df = load_and_clean_data(inst_type)
        if df is None: continue
        
        model, preprocessor = load_model_and_preprocessor(model_name, inst_type)
        if model is None: continue

        # --- CENÁRIO BASE ---
        try:
            X_base = preprocessor.transform(df)
            y_base_pred = model.predict(X_base)
            avg_base = np.mean(y_base_pred)
            print(f"  Média Taxa Integralização (Base): {avg_base:.4f}")
        except Exception as e:
            print(f"  Erro na predição base: {e}")
            continue

        # --- CENÁRIO 1: Expansão Total do Apoio Social ---
        df_apoio = df.copy()
        df_apoio['in_apoio_social'] = 1.0 
        X_apoio = preprocessor.transform(df_apoio)
        y_apoio_pred = model.predict(X_apoio)
        avg_apoio = np.mean(y_apoio_pred)
        delta_apoio = avg_apoio - avg_base
        print(f"  Cenário 'Apoio Social Universal': {avg_apoio:.4f} (Delta: {delta_apoio:+.4f})")

        # --- CENÁRIO 2: Expansão Total do Financiamento ---
        df_fin = df.copy()
        df_fin['in_financiamento_estudantil'] = 1.0
        X_fin = preprocessor.transform(df_fin)
        y_fin_pred = model.predict(X_fin)
        avg_fin = np.mean(y_fin_pred)
        delta_fin = avg_fin - avg_base
        print(f"  Cenário 'Financiamento Universal': {avg_fin:.4f} (Delta: {delta_fin:+.4f})")

        # --- CENÁRIO 3: Intervenção Focada (Grupo de Risco) ---
        mask_risk = y_base_pred < 0.5
        if mask_risk.sum() > 0:
            df_risk_interv = df.copy()
            df_risk_interv.loc[mask_risk, 'in_apoio_social'] = 1.0
            X_risk = preprocessor.transform(df_risk_interv)
            y_risk_pred = model.predict(X_risk)
            avg_risk_before = np.mean(y_base_pred[mask_risk])
            avg_risk_after = np.mean(y_risk_pred[mask_risk])
            delta_risk = avg_risk_after - avg_risk_before
            print(f"  Cenário 'Foco no Risco' (N={mask_risk.sum()}): Delta: {delta_risk:+.4f}")
            risk_uplift = delta_risk
        else:
            risk_uplift = 0.0
            print("  Nenhum aluno no grupo de risco encontrado.")

        results_summary.append({
            'Instituicao': inst_type.upper(),
            'Baseline': avg_base,
            'Apoio_Universal': avg_apoio,
            'Delta_Apoio': delta_apoio,
            'Financ_Universal': avg_fin,
            'Delta_Financ': delta_fin,
            'Uplift_Grupo_Risco': risk_uplift
        })

        # Visualização
        scenarios = ['Base', 'Apoio Total', 'Financ. Total']
        values = [avg_base, avg_apoio, avg_fin]
        plt.figure(figsize=(8, 6))
        bars = plt.bar(scenarios, values, color=['#e0e0e0', '#2c7fb8', '#41b6c4'])
        plt.ylim(min(values)*0.9, max(values)*1.05)
        plt.title(f'Simulação de Impacto de Políticas - {inst_type.upper()}', fontsize=14)
        plt.ylabel('Taxa de Integralização Média Prevista')
        for bar in bars:
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height(), f'{bar.get_height():.3f}', ha='center', va='bottom', fontweight='bold')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'policy_impact_{inst_type}_{model_name}.png'))
        plt.close()

    if results_summary:
        pd.DataFrame(results_summary).to_csv(os.path.join(REPORTS_PATH, 'policy_simulation_results.csv'), index=False)
        print(f"\nResumo das simulações salvo em: {REPORTS_PATH}/policy_simulation_results.csv")

# =============================================================================
# ANÁLISE SHAP
# =============================================================================

def analyze_model_with_shap(model_name):
    """Executa análise SHAP completa."""
    print(f"\n\n--- INICIANDO ANÁLISE DE INTERPRETABILIDADE (SHAP) PARA: {model_name} ---")
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'shap')
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    for inst_type in ['publica', 'privada']:
        print(f"\nAnalisando modelo para IES: {inst_type.upper()}")
        
        model, preprocessor = load_model_and_preprocessor(model_name, inst_type)
        if model is None: continue

        X_test_cleaned = load_and_clean_data(inst_type)
        if X_test_cleaned is None: continue

        if len(X_test_cleaned) > 500:
            X_test_cleaned = X_test_cleaned.sample(500, random_state=42)

        print("Transformando dados...")
        X_test_processed = preprocessor.transform(X_test_cleaned)
        
        if hasattr(X_test_processed, "toarray"):
             X_test_processed_dense = X_test_processed.toarray()
        else:
             X_test_processed_dense = X_test_processed

        try:
            feature_names = preprocessor.get_feature_names_out()
        except:
            feature_names = [f'feat_{i}' for i in range(X_test_processed_dense.shape[1])]

        X_test_processed_df = pd.DataFrame(X_test_processed_dense, columns=feature_names)

        if model_name in ['RandomForest', 'LightGBM', 'GradientBoosting']:
            explainer = shap.TreeExplainer(model)
        else:
            kmeans_summary = shap.kmeans(X_test_processed_dense, 10)
            explainer = shap.KernelExplainer(model.predict, kmeans_summary)

        print("Calculando valores SHAP (pode demorar)...")
        shap_values = explainer.shap_values(X_test_processed_df) 

        if isinstance(shap_values, list):
            shap_values = shap_values[1]

        plt.figure(figsize=(16, 10))
        shap.summary_plot(shap_values, X_test_processed_df, plot_type="bar", show=False)
        plt.title(f'Importância Global (SHAP) - {inst_type.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_bar_{inst_type}_{model_name}.png'))
        plt.close()

        plt.figure(figsize=(16, 10))
        shap.summary_plot(shap_values, X_test_processed_df, show=False)
        plt.title(f'Impacto Detalhado (SHAP) - {inst_type.upper()}')
        plt.tight_layout()
        plt.savefig(os.path.join(FIGURES_PATH, f'shap_summary_beeswarm_{inst_type}_{model_name}.png'))
        plt.close()
        print(f"-> Gráficos SHAP salvos em: {FIGURES_PATH}")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ferramenta de Análise de Previsões e Cenários.')
    
    parser.add_argument(
        '--model', 
        type=str, 
        default='RandomForest',
        choices=['RandomForest', 'LightGBM', 'GradientBoosting', 'SVR', 'Ridge'],
        help='Modelo a ser utilizado.'
    )
    
    parser.add_argument(
        '--analysis', 
        type=str, 
        default='policy', 
        choices=['exhaustive', 'shap', 'policy'],
        help='Tipo de análise: "policy" (Simulação de Intervenção), "exhaustive" (Cenários Sintéticos), "shap" (Interpretabilidade).'
    )
    
    args = parser.parse_args()

    print(f"=== Executando Análise: {args.analysis.upper()} (Modelo: {args.model}) ===")

    if args.analysis == 'shap':
        analyze_model_with_shap(args.model)
    
    elif args.analysis == 'exhaustive':
        csv_path = run_and_save_all_scenarios(args.model)
        analyze_predictions(csv_path)
        
    elif args.analysis == 'policy':
        simulate_policy_interventions(args.model)
    
    print("\n=== Concluído ===")