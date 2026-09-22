"""
tests/test_etl.py
Pruebas unitarias automáticas con pytest para las transformaciones críticas del ETL.
"""
import numpy as np
import pandas as pd
from src.etl import preprocess_and_aggregate


def test_rounding_rules_and_revenue_calculation():
    """Verifica que el redondeo de llamadas y de internet cumplan las especificaciones."""
    users_mock = pd.DataFrame(
        {
            "user_id": [1, 2],
            "city": ["Seattle, WA", "New York, NY"],
            "plan": ["surf", "ultimate"],
            "reg_date": ["2018-01-01", "2018-01-01"],
            "churn_date": [np.nan, np.nan],
        }
    )

    # Usuario 1 (Surf): 2 llamadas de 0.1 min -> deben redondearse a 1 min c/u = 2 min
    calls_mock = pd.DataFrame(
        {
            "id": ["c1", "c2"],
            "user_id": [1, 1],
            "call_date": ["2018-08-05", "2018-08-10"],
            "duration": [0.1, 0.1],
        }
    )

    messages_mock = pd.DataFrame(
        {"id": ["m1"], "user_id": [1], "message_date": ["2018-08-05"]}
    )

    # Usuario 1 (Surf): 15361 MB consumidos en el mes -> límite 15360 MB (15 GB)
    # 15361 / 1024 = 15.0009 -> ceil = 16 GB -> Excedente de 1 GB ($10 USD)
    internet_mock = pd.DataFrame(
        {"id": ["i1"], "user_id": [1], "session_date": ["2018-08-05"], "mb_used": [15361.0]}
    )

    plans_mock = pd.DataFrame(
        {
            "plan": ["surf", "ultimate"],
            "usd_monthly_pay": [20.0, 70.0],
            "minutes_included": [500, 3000],
            "messages_included": [50, 1000],
            "mb_per_month_included": [15360, 30720],
            "usd_per_minute": [0.03, 0.01],
            "usd_per_message": [0.03, 0.01],
            "usd_per_gb": [10.0, 7.0],
        }
    )

    result = preprocess_and_aggregate(
        users_mock, calls_mock, messages_mock, internet_mock, plans_mock
    )

    # Filtrar resultado del usuario 1 en mes 8
    u1 = result[(result["user_id"] == 1) & (result["month"] == 8)].iloc[0]

    assert u1["minutes_spent"] == 2
    assert u1["gb_used"] == 16
    assert u1["cost_extra_internet"] == 10.0
    # Total = Cuota base ($20) + Excedente internet ($10) = $30 USD
    assert u1["total_revenue"] == 30.0