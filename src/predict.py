import joblib
import pandas as pd
import os
import itertools
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import argparse 
import shap 
import json

# Importações do projeto para regenerar clusters
from preprocessing.preprocessor import preprocess_for_kmeans
from modeling.train import predict_clusters

# =============================================================================
# FUNÇÕES AUXILIARES DE CARGA E PREPARAÇÃO
# =============================================================================

def load_model_and_preprocessor(model_name, institution_type):
    """
    Carrega o modelo e o pré-processador salvos.
    """
    MODELS_PATH = 'models'
    path_main_model = os.path.join(MODELS_PATH, f'permanencia_model_{institution_type}.joblib')
    path_test_model = os.path.join(MODELS_PATH, f'{model_name}_{institution_type}_best.joblib')
    preprocessor_path = os.path.join(MODELS_PATH, f'preprocessor_{institution_type}.joblib')

    model_path = None
    if os.path.exists(path_main_model):
        model_path = path_main_model
    elif os.path.exists(path_test_model):
        model_path = path_test_model
        print(f"  [Aviso] Usando modelo de teste: {os.path.basename(model_path)}")
    else:
        print(f"AVISO: Nenhum modelo encontrado para '{institution_type}'.")
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
    path = f'models/kmeans_model_{institution_type}.joblib'
    if os.path.exists(path):
        return joblib.load(path)
    return None

def load_and_clean_data(institution_type):
    data_path = f'data/{institution_type}_sample.csv'
    if not os.path.exists(data_path):
        print(f"ERRO: Arquivo '{data_path}' não encontrado.")
        return None
    df = pd.read_csv(data_path)
    
    # Tratamento de tipos
    if 'tp_sexo' in df.columns:
        df['tp_sexo'] = df['tp_sexo'].astype(object)
        df.loc[df['tp_sexo'].isin([1, '1', '1.0']), 'tp_sexo'] = True
        df.loc[df['tp_sexo'].isin([2, '2', '2.0']), 'tp_sexo'] = False
        df['tp_sexo'] = df['tp_sexo'].astype(bool)
    
    # Regenerar clusters
    kmeans = get_kmeans_model(institution_type)
    if kmeans:
        try:
            X_kmeans, _ = preprocess_for_kmeans(df.copy())
            labels = predict_clusters(kmeans, X_kmeans)
            df['cluster'] = labels
            df['cluster'] = df['cluster'].astype('category')
        except Exception:
            pass
    return df

# =============================================================================
# PREVISÃO E CENÁRIOS
# =============================================================================

def run_and_save_all_scenarios(model_name):
    """(ANÁLISE EXAUSTIVA) Gera cenários preenchendo colunas faltantes com valores padrão."""
    print(f"--- INICIANDO TESTE EXAUSTIVO DE TODOS OS CENÁRIOS ({model_name}) ---")

    possible_values = {
        'tp_cor_raca': {0: "Não declarado", 1: "Branca", 2: "Preta", 3: "Parda", 4: "Amarela", 5: "Indígena"},
        'tp_sexo': {1: "Masculino", 2: "Feminino"},
        'tp_escola_conclusao_ens_medio': {1: "Privada", 2: "Pública"},
        'tp_modalidade_ensino': {1: "Presencial", 2: "EAD"},
        'in_financiamento_estudantil': {0: "Não", 1: "Sim"},
        'in_apoio_social': {0: "Não", 1: "Sim"}
    }
    
    igc_faixas = {1: 0.5, 3: 2.5, 5: 4.5} # Reduzido para agilizar
    
    # Valores padrão para colunas que não estamos variando mas o modelo exige
    base_values_template = {
        'faixa_etaria': 3.0, # 22-25 anos
        'tp_grau_academico': 1.0, # Bacharelado
        'nu_carga_horaria': 3000,
        'duracao_ideal_anos': 4.0,
        'pib': 500000000, # Valor médio genérico
        'inscritos_por_vaga': 5.0,
        'sigla_uf_curso': 'SP',
        'nm_categoria': 'Direito',
        'no_regiao_ies': 'Sudeste',
        'nu_ano_censo': 2019 # Apenas para constar
    }

    keys = possible_values.keys()
    value_codes = [list(v.keys()) for v in possible_values.values()]
    all_combinations = list(itertools.product(*value_codes))
    
    # Pré-carregar recursos
    models_cache = {}
    kmeans_cache = {}
    total_scenarios = 0
    
    for inst_type in ['publica', 'privada']:
        models_cache[inst_type] = load_model_and_preprocessor(model_name, inst_type)
        kmeans_cache[inst_type] = get_kmeans_model(inst_type)
        n_clusters = kmeans_cache[inst_type].n_clusters if kmeans_cache[inst_type] else 1
        total_scenarios += len(all_combinations) * len(igc_faixas) * n_clusters

    print(f"Total de cenários estimados: {total_scenarios}")
    results = []
    count = 0
    errors_logged = 0

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
                    if count % 1000 == 0: print(f"  -> Calculando {count}...", end='\r')

                    profile = student_profile_base.copy()
                    profile['tp_categoria_administrativa'] = inst_code
                    
                    if kmeans:
                        profile['cluster'] = cluster_id
                    
                    df_temp = pd.DataFrame([profile])
                    
                    # Ajustes de tipos críticos
                    df_temp['tp_sexo'] = df_temp['tp_sexo'].map({1: False, 2: True}).astype(bool)
                    for col in ['tp_cor_raca', 'tp_escola_conclusao_ens_medio', 'tp_modalidade_ensino', 
                                'tp_grau_academico', 'sigla_uf_curso', 'nm_categoria', 
                                'tp_categoria_administrativa', 'no_regiao_ies']:
                        df_temp[col] = df_temp[col].astype(str).astype('object')
                    
                    if kmeans:
                        df_temp['cluster'] = df_temp['cluster'].astype('category')

                    try:
                        processed = preprocessor.transform(df_temp)
                        pred = model.predict(processed)[0]
                        
                        # Salvar resultado
                        diferenca = pred - profile['duracao_ideal_anos']
                        status = 'Evasão Provável' if diferenca < -0.5 else 'Atraso' if diferenca > 0.5 else 'Conclusão no Prazo'
                        
                        res_dict = {
                            "Tipo IES": inst_type.upper(),
                            "Cor/Raça": possible_values['tp_cor_raca'][profile['tp_cor_raca']],
                            "Sexo": possible_values['tp_sexo'][profile['tp_sexo']],
                            "Escola Média": possible_values['tp_escola_conclusao_ens_medio'][profile['tp_escola_conclusao_ens_medio']],
                            "Modalidade": possible_values['tp_modalidade_ensino'][profile['tp_modalidade_ensino']],
                            "Financiamento": possible_values['in_financiamento_estudantil'][profile['in_financiamento_estudantil']],
                            "Apoio Social": possible_values['in_apoio_social'][profile['in_apoio_social']],
                            "Faixa IGC": igc_faixa,
                            "Previsão (anos)": round(pred, 2),
                            "Status Conclusão": status
                        }
                        results.append(res_dict)

                    except Exception as e:
                        if errors_logged < 5:
                            print(f"\nErro no cenário: {e}")
                            errors_logged += 1
                        continue

    if not results:
        print("\nERRO CRÍTICO: Nenhum cenário foi calculado com sucesso.")
        return None

    results_df = pd.DataFrame(results)
    REPORTS_PATH = 'reports'
    os.makedirs(REPORTS_PATH, exist_ok=True)
    output_path = os.path.join(REPORTS_PATH, f'prediction_scenarios_{model_name}.csv')
    results_df.to_csv(output_path, index=False)
    print(f"\n--- {len(results)} cenários salvos em: {output_path} ---")
    return output_path

def analyze_predictions(csv_path):
    if not csv_path or not os.path.exists(csv_path): return
    df = pd.read_csv(csv_path)
    if df.empty: 
        print("CSV vazio.")
        return

    FIGURES_PATH = 'reports/figures'
    os.makedirs(FIGURES_PATH, exist_ok=True)
    model_name = os.path.basename(csv_path).replace('prediction_scenarios_', '').replace('.csv', '')

    try:
        plt.figure(figsize=(12, 7))
        sns.countplot(x='Status Conclusão', hue='Tipo IES', data=df, palette='mako')
        plt.title(f'Distribuição Prevista do Status ({model_name})')
        plt.savefig(os.path.join(FIGURES_PATH, f'impacto_status_{model_name}.png'))
        plt.close()

        features = ["Modalidade", "Escola Média", "Financiamento", "Apoio Social", "Cor/Raça"]
        for feat in features:
            if feat not in df.columns: continue
            plt.figure(figsize=(12, 8))
            sns.barplot(x=feat, y='Previsão (anos)', hue='Tipo IES', data=df, palette='viridis')
            plt.title(f'Impacto: {feat}')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(os.path.join(FIGURES_PATH, f'impacto_{feat.lower().replace("/","")}_{model_name}.png'))
            plt.close()
    except Exception as e:
        print(f"Erro ao gerar gráficos: {e}")

# =============================================================================
# SIMULAÇÃO DE POLÍTICAS (Mantida e Simplificada)
# =============================================================================

def simulate_policy_interventions(model_name):
    print(f"\n--- SIMULAÇÃO DE POLÍTICAS ({model_name}) ---")
    REPORTS_PATH = 'reports'
    FIGURES_PATH = os.path.join(REPORTS_PATH, 'figures', 'policy_simulation')
    os.makedirs(FIGURES_PATH, exist_ok=True)
    
    summary = []
    for inst_type in ['publica', 'privada']:
        df = load_and_clean_data(inst_type)
        model, preprocessor = load_model_and_preprocessor(model_name, inst_type)
        if df is None or model is None: continue

        try:
            X_base = preprocessor.transform(df)
            base_mean = model.predict(X_base).mean()
            
            # Apoio Social
            df_apoio = df.copy()
            df_apoio['in_apoio_social'] = 1.0
            apoio_mean = model.predict(preprocessor.transform(df_apoio)).mean()
            
            # Financiamento
            df_fin = df.copy()
            df_fin['in_financiamento_estudantil'] = 1.0
            fin_mean = model.predict(preprocessor.transform(df_fin)).mean()

            print(f"{inst_type.upper()} -> Base: {base_mean:.2f} | Apoio: {apoio_mean:.2f} | Fin: {fin_mean:.2f}")
            
            summary.append({
                'Instituicao': inst_type, 'Base': base_mean, 
                'Apoio_Universal': apoio_mean, 'Financ_Universal': fin_mean
            })
            
            # Plot
            plt.figure(figsize=(6,5))
            plt.bar(['Base', 'Apoio', 'Financ'], [base_mean, apoio_mean, fin_mean], color=['gray', 'blue', 'green'])
            plt.title(f'Impacto Políticas - {inst_type.upper()}')
            plt.ylim(min(base_mean, apoio_mean, fin_mean)*0.95, max(base_mean, apoio_mean, fin_mean)*1.05)
            plt.savefig(os.path.join(FIGURES_PATH, f'policy_{inst_type}.png'))
            plt.close()
            
        except Exception as e:
            print(f"Erro na simulação {inst_type}: {e}")

    if summary:
        pd.DataFrame(summary).to_csv(os.path.join(REPORTS_PATH, 'policy_simulation.csv'), index=False)

# =============================================================================
# SHAP (Mantido)
# =============================================================================

def analyze_model_with_shap(model_name):
    print(f"\n--- ANÁLISE SHAP ({model_name}) ---")
    FIGURES_PATH = 'reports/figures/shap'
    os.makedirs(FIGURES_PATH, exist_ok=True)

    for inst_type in ['publica', 'privada']:
        model, preprocessor = load_model_and_preprocessor(model_name, inst_type)
        df = load_and_clean_data(inst_type)
        if df is None or model is None: continue
        
        if len(df) > 300: df = df.sample(300, random_state=42)
        
        try:
            X_proc = preprocessor.transform(df)
            if hasattr(X_proc, "toarray"): X_proc = X_proc.toarray()
            
            # Nomes das features
            try:
                feats = preprocessor.get_feature_names_out()
            except:
                feats = [f'F{i}' for i in range(X_proc.shape[1])]
            
            X_df = pd.DataFrame(X_proc, columns=feats)
            
            explainer = shap.TreeExplainer(model) if model_name in ['RandomForest', 'LightGBM'] else shap.KernelExplainer(model.predict, X_df)
            shap_values = explainer.shap_values(X_df)
            
            if isinstance(shap_values, list): shap_values = shap_values[1]

            plt.figure()
            shap.summary_plot(shap_values, X_df, show=False)
            plt.savefig(os.path.join(FIGURES_PATH, f'shap_{inst_type}.png'), bbox_inches='tight')
            plt.close()
            print(f"Gráfico SHAP salvo para {inst_type}")
        except Exception as e:
            print(f"Erro no SHAP {inst_type}: {e}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', type=str, default='RandomForest')
    parser.add_argument('--analysis', type=str, default='policy', choices=['exhaustive', 'shap', 'policy'])
    args = parser.parse_args()

    if args.analysis == 'shap': analyze_model_with_shap(args.model)
    elif args.analysis == 'exhaustive': 
        csv = run_and_save_all_scenarios(args.model)
        analyze_predictions(csv)
    elif args.analysis == 'policy': simulate_policy_interventions(args.model)