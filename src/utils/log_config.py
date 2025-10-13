import logging
import os
from pathlib import Path
from datetime import datetime

# Define a pasta de logs padrão
LOG_DIR = Path('logs')

def setup_logging(log_filename="pipeline_execution.log", 
                  log_level=logging.INFO, 
                  stream_level=logging.INFO,
                  enable_logging=True):
    """
    Configura o sistema de logging: cria o diretório de logs se não existir
    """
    logger = logging.getLogger() # Pega o logger raiz

    if not enable_logging:
        # Define o nível do logger raiz para CRITICAL. 
        # Isso garante que a maioria dos INFO/DEBUG/WARNING seja ignorada, 
        # minimizando a sobrecarga de performance.
        logger.setLevel(logging.CRITICAL)
        return logger
    # ----------------------------------------------------
    # 1. Cria o diretório de logs se ele não existir
    # os.makedirs(..., exist_ok=True) é a forma segura e recomendada
    os.makedirs(LOG_DIR, exist_ok=True)
    
    log_path = LOG_DIR / log_filename
    
    # 2. Configuração do Formato
    # Formato detalhado para o arquivo de log
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    # Formato mais conciso para o console
    stream_formatter = logging.Formatter('%(levelname)s: %(message)s')

    # 3. Criação do Logger Raiz
    # Sempre configure o logger do módulo/pacote principal (ou o root logger)
    logger = logging.getLogger() # Pega o logger raiz
    logger.setLevel(logging.DEBUG) # Nível mais baixo para garantir que todos os logs sejam capturados

    # Evita adicionar handlers se a função for chamada mais de uma vez
    if not logger.handlers:
        
        # 4. Configuração do Handler para Arquivo (FileHandler)
        file_handler = logging.FileHandler(log_path, mode='w', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(file_formatter)
        logger.addHandler(file_handler)

        # 5. Configuração do Handler para Console (StreamHandler)
        stream_handler = logging.StreamHandler()
        stream_handler.setLevel(stream_level)
        stream_handler.setFormatter(stream_formatter)
        logger.addHandler(stream_handler)

    logger.info(f"Sistema de logging configurado. Logs salvos em: {log_path.resolve()}")
    
    return logger

# Exemplo de como gerar um nome de arquivo único baseado na data (opcional)
def get_dated_log_filename(prefix="pipeline", extension=".log"):
    """Gera um nome de arquivo de log com timestamp."""
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}{extension}"