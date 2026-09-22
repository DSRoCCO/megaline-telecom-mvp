"""
src/etl.py
Pipeline de Extracción, Transformación y Carga (ETL) para Megaline Telecom.
"""
import numpy as np
import pandas as pd
from src.config import (
    USERS_CSV,
    CALLS_CSV,
    MESSAGES_CSV,
    INTERNET_CSV,
    PLANS_CSV,
    PROCESSED_DATA_PATH,
    MB_PER_GB,
)


def load_raw_data():
    """Carga los 5 archivos CSV fuente desde data/raw."""
    df_users = pd.read_csv(USERS_CSV)
    df_calls = pd.read_csv(CALLS_CSV)
    df_messages = pd.read_csv(MESSAGES_CSV)
    df_internet = pd.read_csv(INTERNET_CSV)
    df_plans = pd.read_csv(PLANS_CSV)
    return df_users, df_calls, df_messages, df_internet, df_plans


def preprocess_and_aggregate(df_users, df_calls, df_messages, df_internet, df_plans):
    """
    Aplica redondeos, transformaciones temporales, agregaciones mensuales
    y cálculo vectorizado de ingresos según la regla de negocio.
    """
    # 1. Transformación de fechas
    df_users["reg_date"] = pd.to_datetime(df_users["reg_date"])
    df_users["churn_date"] = pd.to_datetime(df_users["churn_date"])
    df_calls["call_date"] = pd.to_datetime(df_calls["call_date"])
    df_messages["message_date"] = pd.to_datetime(df_messages["message_date"])
    df_internet["session_date"] = pd.to_datetime(df_internet["session_date"])

    # 2. Extracción de mes
    df_calls["month"] = df_calls["call_date"].dt.month
    df_messages["month"] = df_messages["message_date"].dt.month
    df_internet["month"] = df_internet["session_date"].dt.month

    # 3. Regla de llamadas: redondeo individual hacia arriba
    df_calls["duration_rounded"] = np.ceil(df_calls["duration"]).astype(int)

    # 4. Agregaciones mensuales por usuario y mes
    calls_monthly = (
        df_calls.groupby(["user_id", "month"])
        .agg(calls_count=("id", "count"), minutes_spent=("duration_rounded", "sum"))
        .reset_index()
    )

    messages_monthly = (
        df_messages.groupby(["user_id", "month"])
        .agg(messages_sent=("id", "count"))
        .reset_index()
    )

    internet_monthly = (
        df_internet.groupby(["user_id", "month"])
        .agg(mb_used=("mb_used", "sum"))
        .reset_index()
    )
    # Regla de internet: redondeo del acumulado mensual a GB
    internet_monthly["gb_used"] = np.ceil(internet_monthly["mb_used"] / MB_PER_GB).astype(int)

    # 5. Fusión Outer para conservar meses con uso parcial
    df_monthly = calls_monthly.merge(messages_monthly, on=["user_id", "month"], how="outer")
    df_monthly = df_monthly.merge(internet_monthly, on=["user_id", "month"], how="outer")

    # Imputación de ceros en consumos ausentes
    consumption_cols = ["calls_count", "minutes_spent", "messages_sent", "mb_used", "gb_used"]
    df_monthly[consumption_cols] = df_monthly[consumption_cols].fillna(0)

    # Fusión con datos demográficos del usuario
    df_monthly = df_monthly.merge(
        df_users[["user_id", "city", "plan", "churn_date"]],
        on="user_id",
        how="left",
    )

    # 6. Cruce con condiciones tarifarias
    if "plan_name" in df_plans.columns:
        df_plans = df_plans.rename(columns={"plan_name": "plan"})

    df_monthly = df_monthly.merge(df_plans, on="plan", how="left")

    # 7. Cálculo vectorizado de excedentes e ingresos
    extra_minutes = np.maximum(0, df_monthly["minutes_spent"] - df_monthly["minutes_included"])
    extra_messages = np.maximum(0, df_monthly["messages_sent"] - df_monthly["messages_included"])
    extra_gb = np.maximum(
        0,
        df_monthly["gb_used"] - np.ceil(df_monthly["mb_per_month_included"] / MB_PER_GB),
    )

    df_monthly["cost_extra_calls"] = extra_minutes * df_monthly["usd_per_minute"]
    df_monthly["cost_extra_messages"] = extra_messages * df_monthly["usd_per_message"]
    df_monthly["cost_extra_internet"] = extra_gb * df_monthly["usd_per_gb"]
    df_monthly["overage_revenue"] = (
        df_monthly["cost_extra_calls"]
        + df_monthly["cost_extra_messages"]
        + df_monthly["cost_extra_internet"]
    )

    df_monthly["total_revenue"] = df_monthly["usd_monthly_pay"] + df_monthly["overage_revenue"]

    # Target binario: 1 para ultimate, 0 para surf
    df_monthly["target_plan"] = (df_monthly["plan"] == "ultimate").astype(int)

    return df_monthly


def run_etl():
    """Ejecuta el pipeline completo de ETL y guarda el resultado en Parquet."""
    print("Iniciando pipeline ETL de Megaline...")
    df_u, df_c, df_m, df_i, df_p = load_raw_data()
    df_clean = preprocess_and_aggregate(df_u, df_c, df_m, df_i, df_p)
    df_clean.to_parquet(PROCESSED_DATA_PATH, index=False)
    print(f"ETL finalizado con éxito. Dimensiones: {df_clean.shape}")
    print(f"Dataset guardado en: {PROCESSED_DATA_PATH}")
    return df_clean


if __name__ == "__main__":
    run_etl()