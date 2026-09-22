"""
src/app.py
Dashboard interactivo para Megaline Telecom: Rentabilidad, Hipótesis, MLOps y Upselling.
Incluye visualizaciones interactivas con Plotly y gestión de estado con st.session_state.
"""
from pathlib import Path
import sys

ROOT_DIR = Path(__file__).resolve().parent.parent
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
import streamlit as st

from src.config import PROCESSED_DATA_PATH
from src.models import FEATURE_COLS, build_models, get_data_splits
from src.stats_tests import test_geography_revenue, test_plans_revenue

st.set_page_config(page_title="Megaline Telecom - Decision System", layout="wide")


@st.cache_data
def load_data():
    return pd.read_parquet(PROCESSED_DATA_PATH)


try:
    df = load_data()
except Exception as e:
    st.error(f"Error cargando dataset procesado: {e}")
    st.stop()

# ----------------- INICIALIZACIÓN DE VALORES POR DEFECTO -----------------
DEFAULTS = {
    # Pestaña 1
    "tab1_plans": ["surf", "ultimate"],
    "tab1_metric": "minutes_spent",
    # Pestaña 2
    "tab2_alpha": 0.05,
    # Pestaña 3
    "tab3_model": "Random Forest (Optimizado Accuracy)",
    # Pestaña 4
    "tab4_threshold": 70,
    "tab4_calls": 60,
    "tab4_minutes": 450,
    "tab4_sms": 30,
    "tab4_gb": 22,
}

for key, val in DEFAULTS.items():
    if key not in st.session_state:
        st.session_state[key] = val


# Funciones de reinicio por pestaña
def reset_tab1():
    st.session_state.tab1_plans = DEFAULTS["tab1_plans"]
    st.session_state.tab1_metric = DEFAULTS["tab1_metric"]


def reset_tab2():
    st.session_state.tab2_alpha = DEFAULTS["tab2_alpha"]


def reset_tab3():
    st.session_state.tab3_model = DEFAULTS["tab3_model"]


def reset_tab4():
    st.session_state.tab4_threshold = DEFAULTS["tab4_threshold"]
    st.session_state.tab4_calls = DEFAULTS["tab4_calls"]
    st.session_state.tab4_minutes = DEFAULTS["tab4_minutes"]
    st.session_state.tab4_sms = DEFAULTS["tab4_sms"]
    st.session_state.tab4_gb = DEFAULTS["tab4_gb"]


# ----------------- CABECERA Y KPIS -----------------
st.title("📱 Megaline Telecom: Sistema de Soporte a Decisiones Comerciales")
st.markdown(
    "Herramienta interactiva para análisis de consumo, validación estadística y recomendación de tarifas."
)

m1, m2, m3, m4 = st.columns(4)
total_rev = df["total_revenue"].sum()
surf_rev_pct = (df[df["plan"] == "surf"]["total_revenue"].sum() / total_rev) * 100
arpu_surf = df[df["plan"] == "surf"]["total_revenue"].mean()
arpu_ult = df[df["plan"] == "ultimate"]["total_revenue"].mean()

m1.metric("Ingresos Totales", f"${total_rev:,.0f} USD")
m2.metric(
    "Facturación Surf",
    f"{surf_rev_pct:.1f}%",
    help="Porcentaje del ingreso total aportado por Surf",
)
m3.metric("ARPU Surf", f"${arpu_surf:.2f} USD")
m4.metric("ARPU Ultimate", f"${arpu_ult:.2f} USD", delta=f"{arpu_ult - arpu_surf:+.2f} USD")

# Preparar particiones de datos para ML
X_train, X_test, y_train, y_test = get_data_splits(df)

tab1, tab2, tab3, tab4 = st.tabs(
    [
        "📊 Explorador Dinámico de Consumos",
        "🧪 Contraste de Hipótesis",
        "🤖 Diagnóstico de Modelos ML",
        "🎯 Churn & Simulador de Upselling",
    ]
)

# ----------------- TAB 1: EDA INTERACTIVO -----------------
with tab1:
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.header("Análisis de Consumo e Ingresos")
    with reset_col:
        st.button("🔄 Restablecer Filtros", on_click=reset_tab1, key="btn_reset_tab1")

    col_filter, col_plot = st.columns([1, 2])

    with col_filter:
        st.subheader("Controles de Filtrado")
        selected_plans = st.multiselect(
            "Filtrar Planes:",
            options=["surf", "ultimate"],
            key="tab1_plans",
        )
        metric_choice = st.selectbox(
            "Seleccionar Variable a Evaluar:",
            options=["minutes_spent", "messages_sent", "gb_used", "total_revenue"],
            format_func=lambda x: {
                "minutes_spent": "Minutos de Llamada",
                "messages_sent": "Mensajes Enviados",
                "gb_used": "Datos (GB)",
                "total_revenue": "Ingreso Mensual ($)",
            }[x],
            key="tab1_metric",
        )

        filtered_df = df[df["plan"].isin(selected_plans)] if selected_plans else df

        st.write("### Resumen Estadístico (Filtrado)")
        stats_summary = (
            filtered_df.groupby("plan")[metric_choice]
            .agg(
                Media="mean",
                Mediana="median",
                Desv_Std="std",
                IQR=lambda x: np.percentile(x, 75) - np.percentile(x, 25),
            )
            .round(2)
        )
        st.dataframe(stats_summary)

    with col_plot:
        fig, ax = plt.subplots(figsize=(8, 4.5))
        sns.histplot(
            data=filtered_df,
            x=metric_choice,
            hue="plan",
            kde=True,
            stat="density",
            common_norm=False,
            ax=ax,
        )
        ax.set_title(f"Distribución de {metric_choice}")
        st.pyplot(fig)

# ----------------- TAB 2: HIPÓTESIS INTERACTIVAS -----------------
with tab2:
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.header("Validación de Hipótesis Estadísticas")
    with reset_col:
        st.button("🔄 Restablecer Alpha", on_click=reset_tab2, key="btn_reset_tab2")

    alpha = st.slider(
        "Ajustar Nivel de Significancia (Alpha):",
        min_value=0.01,
        max_value=0.10,
        step=0.01,
        key="tab2_alpha",
    )

    col_h1, col_h2 = st.columns(2)
    with col_h1:
        st.subheader("1. Tarifas: Surf vs. Ultimate")
        h1 = test_plans_revenue(df, alpha=alpha)
        st.write(
            f"- **ARPU Surf:** `${h1['mean_surf']:.2f}` USD | **ARPU Ultimate:** `${h1['mean_ultimate']:.2f}` USD"
        )
        st.write(f"- **Estadístico t (Welch):** `{h1['welch_t']:.4f}`")
        st.write(f"- **p-value:** `{h1['welch_p']:.4e}` (Alpha = `{alpha}`)")
        st.write(f"- **Tamaño del Efecto (d de Cohen):** `{h1['cohen_d']:.4f}`")
        if h1["welch_p"] < alpha:
            st.success("✅ **Conclusión:** Se rechaza H0. Hay diferencia significativa de ingresos entre planes.")
        else:
            st.info("ℹ️ **Conclusión:** No se rechaza H0 para el alpha seleccionado.")

    with col_h2:
        st.subheader("2. Geografía: NY-NJ vs. Otras Regiones")
        h2 = test_geography_revenue(df, alpha=alpha)
        st.write(f"- **ARPU NY-NJ:** `${h2['mean_nynj']:.2f}` USD (n={h2['n_nynj']})")
        st.write(f"- **ARPU Otras Regiones:** `${h2['mean_other']:.2f}` USD (n={h2['n_other']})")
        st.write(f"- **Estadístico t (Welch):** `{h2['welch_t']:.4f}`")
        st.write(f"- **p-value:** `{h2['welch_p']:.4f}`")
        if h2["welch_p"] < alpha:
            st.warning(
                "⚠️ **Conclusión de Negocio:** Diferencia estadísticamente detectable pero con brecha monetaria marginal ($5.30 USD), desaconsejando tarifas regionales."
            )

# ----------------- TAB 3: DIAGNÓSTICO ML INTERACTIVO -----------------
with tab3:
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.header("Evaluación y Diagnóstico de Modelos")
    with reset_col:
        st.button("🔄 Restablecer Modelo", on_click=reset_tab3, key="btn_reset_tab3")

    models = build_models()
    model_names = list(models.keys())

    selected_model_name = st.selectbox(
        "Seleccione el Modelo para Inspección Profunda:",
        model_names,
        key="tab3_model",
    )
    active_model = models[selected_model_name]
    active_model.fit(X_train, y_train)
    y_pred = active_model.predict(X_test)

    c_diag1, c_diag2, c_diag3 = st.columns([1, 1, 1.2])

    with c_diag1:
        st.subheader("Matriz de Confusión")
        cm = confusion_matrix(y_test, y_pred)
        fig_cm, ax_cm = plt.subplots(figsize=(4, 3))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["Surf", "Ultimate"],
            yticklabels=["Surf", "Ultimate"],
            ax=ax_cm,
        )
        st.pyplot(fig_cm)

    with c_diag2:
        st.subheader("Curva ROC")
        if hasattr(active_model, "predict_proba"):
            y_prob = active_model.predict_proba(X_test)[:, 1]
            fpr, tpr, _ = roc_curve(y_test, y_prob)
            auc_score = roc_auc_score(y_test, y_prob)
            fig_roc, ax_roc = plt.subplots(figsize=(4, 3))
            ax_roc.plot(fpr, tpr, label=f"AUC = {auc_score:.2f}")
            ax_roc.plot([0, 1], [0, 1], "k--")
            ax_roc.legend()
            st.pyplot(fig_roc)
        else:
            st.write("Este estimador base no cuenta con probabilidades predictivas.")

    with c_diag3:
        st.subheader("Rendimiento por Tarifa")
        rep = classification_report(
            y_test,
            y_pred,
            target_names=["Surf", "Ultimate"],
            zero_division=0,
            output_dict=True,
        )
        df_rep = pd.DataFrame(rep).transpose()

        # 1. Tabla exclusiva de las clases reales (Surf y Ultimate)
        df_classes = df_rep.loc[["Surf", "Ultimate"]].copy()
        df_classes["support"] = df_classes["support"].astype(int)
        st.dataframe(
            df_classes.style.format(
                {
                    "precision": "{:.2f}",
                    "recall": "{:.2f}",
                    "f1-score": "{:.2f}",
                    "support": "{:d}",
                }
            )
        )

        # 2. Métricas globales consolidadas en tarjetas
        st.subheader("Métricas Globales")
        col_m1, col_m2 = st.columns(2)
        col_m1.metric("Accuracy General", f"{rep['accuracy']:.1%}")
        col_m2.metric("F1 Ponderado", f"{rep['weighted avg']['f1-score']:.2f}")

# ----------------- TAB 4: ACCIÓN COMERCIAL, PLOTLY Y UPSELLING -----------------
with tab4:
    header_col, reset_col = st.columns([5, 1])
    with header_col:
        st.header("Gestión de Retención y Recomendación de Planes")
    with reset_col:
        st.button("🔄 Restablecer Valores", on_click=reset_tab4, key="btn_reset_tab4")

    col_churn, col_sim = st.columns([1, 1])

    with col_churn:
        st.subheader("Campaña Preventiva de Fuga (Bill-Shock)")
        surf_avg = (
            df[df["plan"] == "surf"].groupby("user_id")["total_revenue"].mean().reset_index()
        )
        surf_avg.columns = ["user_id", "avg_monthly_revenue"]

        threshold = st.slider(
            "Umbral de gasto mensual para alerta ($ USD):",
            min_value=30,
            max_value=150,
            step=5,
            key="tab4_threshold",
        )

        high_risk_users = surf_avg[surf_avg["avg_monthly_revenue"] > threshold]
        pct_risk = (len(high_risk_users) / len(surf_avg)) * 100

        st.write(f"- **Total Clientes Surf analizados:** {len(surf_avg)}")
        st.write(
            f"- **Clientes que superan ${threshold} USD:** `{len(high_risk_users)}` ({pct_risk:.1f}%)"
        )

        # Gráfico interactivo con Plotly Express usando width="stretch"
        fig_hist = px.histogram(
            surf_avg,
            x="avg_monthly_revenue",
            nbins=30,
            labels={"avg_monthly_revenue": "Gasto Promedio Mensual ($ USD)"},
            title=f"Distribución de Facturación Surf (Corte en > ${threshold} USD)",
            color_discrete_sequence=["#4c72b0"],
        )
        fig_hist.add_vline(
            x=threshold,
            line_dash="dash",
            line_color="red",
            annotation_text=f"Corte: ${threshold} USD",
            annotation_position="top right",
        )
        fig_hist.update_layout(
            yaxis_title="Cantidad de Clientes",
            bargap=0.05,
            margin=dict(l=20, r=20, t=40, b=20),
        )

        st.plotly_chart(fig_hist, width="stretch")
        st.info(
            f"💡 **Acción recomendada:** Contactar a los {len(high_risk_users)} usuarios para proponer la cuota fija Ultimate ($70 USD) y estabilizar el ingreso recurrente."
        )

    with col_sim:
        st.subheader("Simulador de Recomendación en Tiempo Real")
        c_in1, c_in2 = st.columns(2)
        sim_calls = c_in1.slider("Llamadas al mes:", 0, 200, key="tab4_calls")
        sim_minutes = c_in2.slider("Minutos al mes:", 0, 1500, key="tab4_minutes")
        sim_sms = c_in1.slider("SMS al mes:", 0, 200, key="tab4_sms")
        sim_gb = c_in2.slider("GB de Internet al mes:", 0, 60, key="tab4_gb")

        rf_balanced = build_models()["Random Forest (Balanced)"]
        rf_balanced.fit(X_train, y_train)

        sample_input = pd.DataFrame(
            [[sim_calls, sim_minutes, sim_sms, sim_gb]], columns=FEATURE_COLS
        )
        prob = rf_balanced.predict_proba(sample_input)[0][1]

        st.write(f"### Propensión estimada a Ultimate: `{prob:.1%}`")
        if sim_gb > 15 or prob > 0.45:
            st.error(
                "⚠️ **Alerta:** Riesgo inminente de sobrecostos en Surf. Recomendar tarifa Ultimate."
            )
        else:
            st.success("✅ **Óptimo:** El usuario no supera los límites incluidos de Surf.")
