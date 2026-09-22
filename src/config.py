"""
src/config.py
Configuración centralizada de rutas absolutas, constantes de negocio y reproducibilidad.
"""
from pathlib import Path

# Raíz dinámica del proyecto (dos niveles arriba de este archivo)
ROOT_DIR = Path(__file__).resolve().parent.parent

# Directorios del sistema
DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
REPORTS_DIR = ROOT_DIR / "reports"

# Asegurar existencia de directorios de salida
PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Archivos de entrada (data/raw)
USERS_CSV = RAW_DATA_DIR / "megaline_users.csv"
CALLS_CSV = RAW_DATA_DIR / "megaline_calls.csv"
MESSAGES_CSV = RAW_DATA_DIR / "megaline_messages.csv"
INTERNET_CSV = RAW_DATA_DIR / "megaline_internet.csv"
PLANS_CSV = (
    RAW_DATA_DIR / "megaline_plans.csv"
    if (RAW_DATA_DIR / "megaline_plans.csv").exists()
    else RAW_DATA_DIR / "megaline_tariffs.csv"
)

# Archivo de salida procesado
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "megaline_monthly_cleaned.parquet"

# Parámetros de negocio y reproducibilidad
RANDOM_STATE = 42
TEST_SIZE = 0.25
ALPHA_SIGNIFICANCE = 0.05
MB_PER_GB = 1024