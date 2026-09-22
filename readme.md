# Megaline Telecom - Decision Support System MVP 📊

Un Producto Mínimo Viable (MVP) interactivo y contenerizado diseñado como sistema de soporte para la toma de decisiones comerciales en el sector de telecomunicaciones. Integra un pipeline ETL robusto, análisis estadístico de consumo, modelos predictivos de Machine Learning para segmentación de usuarios y un panel analítico en Streamlit.

---

## 🧭 Acceso Rápido al Análisis Exploratorio

Para auditar la exploración estadística inicial, las distribuciones de consumo y las pruebas de hipótesis sin descargar ni compilar el proyecto:

👉 **[Ver Análisis Exploratorio (Jupyter Notebook)](notebook/exploratory_analysis.ipynb)**


## 🏗️ Arquitectura del Proyecto

El repositorio implementa una arquitectura desacoplada para garantizar modularidad, mantenibilidad y cobertura de pruebas:

```text
megaline-telecom-mvp/
├── data/
│   ├── raw/                             # Datasets crudos de telecomunicaciones
│   └── megaline_monthly_cleaned.parquet # Dataset mensual consolidado y procesado
├── notebook/
│   └── exploratory_analysis.ipynb       # Análisis exploratorio y pruebas estadísticas
├── src/
│   ├── config.py                        # Rutas canónicas y parámetros globales
│   ├── etl.py                           # Pipeline de extracción, limpieza y agregación
│   ├── models.py                        # Entrenamiento, validación y métricas de ML
│   ├── stats_tests.py                   # Pruebas de hipótesis estadísticas (t-test / Levene)
│   └── app.py                           # Dashboard interactivo en Streamlit
├── tests/
│   └── test_etl.py                      # Pruebas unitarias sobre transformaciones
├── Dockerfile                           # Definición de contenedor multi-stage con uv
├── .dockerignore                        # Exclusión de contexto de compilación
├── pyproject.toml                       # Gestión centralizada de dependencias y metadatos
└── README.md                            # Documentación técnica

---

## 🚀 Despliegue con Docker

La forma estándar y aislada para ejecutar la aplicación sin requerir configuración previa de Python en el entorno anfitrión.

# 1. Construir la imagen local

docker build -t megaline-telecom-mvp:latest .


# 2. Ejecutar el contenedor

docker run -d -p 8501:8501 --name megaline-app megaline-telecom-mvp:latest


Accede al dashboard interactivo desde tu navegador web en: http://localhost:8501

# 3. Ejecutar pruebas unitarias dentro del contenedor

docker run --rm megaline-telecom-mvp:latest pytest


# 4. Detener y remover el contenedor

docker rm -f megaline-app

---

## 🛠️ Entorno de Desarrollo Local

Si deseas colaborar o trabajar directamente sobre el código fuente utilizando el gestor de dependencias `uv`:

### 1. Inicializar entorno virtual e instalar dependencias
```bash
uv venv
uv pip install -e .

### 2. Correr la suite de pruebas unitarias

pytest


### 3. Iniciar el dashboard localmente

streamlit run src/app.py

---

## 📦 Stack Tecnológico

* **Lenguaje:** Python 3.12
* **Gestión de dependencias:** Astral `uv` / PEP 621 (`pyproject.toml`)
* **Procesamiento de datos y modelado:** Pandas, NumPy, Scikit-Learn, SciPy, PyArrow
* **Interfaz y visualización:** Streamlit, Matplotlib, Seaborn
* **Pruebas y calidad de código:** Pytest, Ruff
* **Contenedorización:** Docker (Debian Slim)

---

## 📄 Licencia

Este proyecto fue desarrollado como parte de un estudio de optimización de ingresos y retención para servicios de telecomunicaciones. Libre para uso analítico y académico.