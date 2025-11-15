import subprocess
import os
import sys
import shutil
from datetime import datetime

# --- Configuração ---

# 1. Lista de valores de K para testar
K_VALUES = [4, 5, 6, 10, 11, 14, 15]

# 2. Nome do seu script principal que faz a análise
TARGET_SCRIPT = "main.py"

# 3. Pasta para salvar os logs de saída
LOG_DIRECTORY = "experiment_logs"

if not os.path.isabs(TARGET_SCRIPT):
    # Assume que main.py está na mesma pasta que este script
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        TARGET_SCRIPT = os.path.join(current_dir, TARGET_SCRIPT)
    except NameError:
        print("Aviso: __file__ não definido. Usando caminho relativo, verifique o diretório de execução.")

def run_all_experiments():
    """
    Executa o script principal de análise para cada valor de K
    e salva o stdout/stderr em arquivos de log individuais.
    """
    print(f"Iniciando execução de {len(K_VALUES)} experimentos...")
    print(f"Logs serão salvos em: {os.path.abspath(LOG_DIRECTORY)}\n")

    # Garante que o diretório de log exista
    os.makedirs(LOG_DIRECTORY, exist_ok=True)
    
    # Limpa logs antigos (opcional, mas recomendado)
    print("Limpando logs antigos...")
    for f in os.listdir(LOG_DIRECTORY):
        os.remove(os.path.join(LOG_DIRECTORY, f))

    # Identifica o executável python (python.exe, python3, etc.)
    python_executable = sys.executable

    start_time_all = datetime.now()

    # Itera por cada valor de K
    for k in K_VALUES:
        k_str = str(k)
        start_time_k = datetime.now()
        
        print(f"--- [k={k}] Iniciando execução... ---")

        # 1. Define o nome do arquivo de log para esta execução
        log_filename = os.path.join(LOG_DIRECTORY, f"log_k_{k:02d}.txt")

        # 2. Monta o comando a ser executado
        command = [
            python_executable,
            TARGET_SCRIPT,
            "-s",
            "--k_privada", k_str,
            "--k_publica", k_str
        ]

        try:
            # 3. Executa o script e captura a saída (stdout e stderr)
            # Removemos 'text=True' e 'encoding' para capturar a saída como BYTES puros.
            process = subprocess.run(
                command,
                capture_output=True,
                check=True  # Faz o Python lançar um erro se o script retornar um código != 0
            )
            
            # 4. Salva a saída no arquivo de log
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"Comando executado: {' '.join(command)}\n")
                f.write(f"Iniciado em: {start_time_k.strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                
                f.write("--- SAÍDA PADRÃO (STDOUT) ---\n\n")
                # Decodifica os bytes para string, substituindo erros por '?'
                f.write(process.stdout.decode('utf-8', errors='replace'))
                
                if process.stderr:
                    f.write("\n\n--- SAÍDA DE ERRO (STDERR) ---\n\n")
                    f.write(process.stderr.decode('utf-8', errors='replace'))
            
            duration_k = (datetime.now() - start_time_k).total_seconds()
            print(f"--- [k={k}] Concluído com sucesso ({duration_k:.2f}s). Log salvo em: {log_filename} ---")

        except subprocess.CalledProcessError as e:
            # Se o script falhar (returncode != 0)
            print(f"--- [k={k}] FALHOU. ---")
            
            # Salva o log de falha
            with open(log_filename, 'w', encoding='utf-8') as f:
                f.write(f"Comando executado: {' '.join(command)}\n")
                f.write(f"FALHOU com código de retorno: {e.returncode}\n")
                f.write("="*80 + "\n\n")
                
                f.write("--- SAÍDA PADRÃO (STDOUT) ---\n\n")
                # TAMBÉM é preciso decodificar aqui, pois 'e.stdout' são bytes
                f.write(e.stdout.decode('utf-8', errors='replace'))
                
                f.write("\n\n--- SAÍDA DE ERRO (STDERR) ---\n\n")
                # E aqui
                f.write(e.stderr.decode('utf-8', errors='replace'))
            
            print(f" Log de erro salvo em: {log_filename}")
            
            # Decodifica a última linha do erro para exibição
            last_error_line = e.stderr.decode('utf-8', errors='replace').strip().splitlines()[-1:]
            print(f"Erro (stderr): {last_error_line}")

        except FileNotFoundError:
            print(f"ERRO CRÍTICO: Não foi possível encontrar o script '{TARGET_SCRIPT}'.")
            print("Verifique se o nome do arquivo 'TARGET_SCRIPT' está correto.")
            return

    duration_all = (datetime.now() - start_time_all).total_seconds()
    print(f"\n--- Todos os {len(K_VALUES)} experimentos concluídos em {duration_all:.2f}s ---")

if __name__ == "__main__":
    run_all_experiments()