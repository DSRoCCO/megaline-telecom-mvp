"""
src/stats_tests.py
Módulo de pruebas estadísticas de hipótesis para rentabilidad y segmentación geográfica.
"""
import numpy as np
import pandas as pd
from scipy import stats
from src.config import PROCESSED_DATA_PATH, ALPHA_SIGNIFICANCE


def cohen_d(x: pd.Series, y: pd.Series) -> float:
    """Calcula el tamaño del efecto d de Cohen entre dos muestras independientes."""
    nx, ny = len(x), len(y)
    dof = nx + ny - 2
    pooled_std = np.sqrt(((nx - 1) * np.var(x, ddof=1) + (ny - 1) * np.var(y, ddof=1)) / dof)
    return float((np.mean(x) - np.mean(y)) / pooled_std)


def test_plans_revenue(df: pd.DataFrame, alpha=ALPHA_SIGNIFICANCE):
    """
    Hipótesis 1: Contraste de ingresos promedio entre Surf y Ultimate.
    Aplica Levene, t de Welch, Mann-Whitney U y d de Cohen.
    """
    rev_surf = df[df["plan"] == "surf"]["total_revenue"]
    rev_ultimate = df[df["plan"] == "ultimate"]["total_revenue"]

    stat_levene, p_levene = stats.levene(rev_surf, rev_ultimate)
    t_stat, p_val_welch = stats.ttest_ind(rev_surf, rev_ultimate, equal_var=False)
    u_stat, p_val_mwu = stats.mannwhitneyu(rev_surf, rev_ultimate)
    effect_size = cohen_d(rev_ultimate, rev_surf)

    results = {
        "mean_surf": float(rev_surf.mean()),
        "mean_ultimate": float(rev_ultimate.mean()),
        "levene_p": float(p_levene),
        "welch_t": float(t_stat),
        "welch_p": float(p_val_welch),
        "mwu_p": float(p_val_mwu),
        "cohen_d": float(effect_size),
        "reject_null": bool(p_val_welch < alpha),
    }
    return results


def test_geography_revenue(df: pd.DataFrame, alpha=ALPHA_SIGNIFICANCE):
    """
    Hipótesis 2: Contraste de ingresos promedio entre NY-NJ y Otras Regiones.
    """
    is_nynj = df["city"].str.contains("NY-NJ", case=False, na=False)
    rev_nynj = df[is_nynj]["total_revenue"]
    rev_other = df[~is_nynj]["total_revenue"]

    t_stat_geo, p_val_geo = stats.ttest_ind(rev_nynj, rev_other, equal_var=False)

    results = {
        "mean_nynj": float(rev_nynj.mean()),
        "n_nynj": int(len(rev_nynj)),
        "mean_other": float(rev_other.mean()),
        "n_other": int(len(rev_other)),
        "welch_t": float(t_stat_geo),
        "welch_p": float(p_val_geo),
        "reject_null": bool(p_val_geo < alpha),
    }
    return results


def run_all_tests():
    """Carga los datos procesados y ejecuta las pruebas de hipótesis."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)

    h1 = test_plans_revenue(df)
    print("\n--- Resultados Hipótesis 1 (Surf vs. Ultimate) ---")
    print(f"Media Surf: ${h1['mean_surf']:.2f} USD | Media Ultimate: ${h1['mean_ultimate']:.2f} USD")
    print(f"Prueba Levene p-value: {h1['levene_p']:.4e}")
    print(f"Prueba t de Welch: t = {h1['welch_t']:.4f}, p-value = {h1['welch_p']:.4e}")
    print(f"Prueba Mann-Whitney U: p-value = {h1['mwu_p']:.4e}")
    print(f"Tamaño del efecto (d de Cohen): {h1['cohen_d']:.4f}")
    print(f"¿Se rechaza H0?: {h1['reject_null']}")

    h2 = test_geography_revenue(df)
    print("\n--- Resultados Hipótesis 2 (NY-NJ vs. Otras Regiones) ---")
    print(f"Media NY-NJ: ${h2['mean_nynj']:.2f} USD (n={h2['n_nynj']})")
    print(f"Media Otras Regiones: ${h2['mean_other']:.2f} USD (n={h2['n_other']})")
    print(f"Prueba t de Welch: t = {h2['welch_t']:.4f}, p-value = {h2['welch_p']:.4f}")
    print(f"¿Se rechaza H0?: {h2['reject_null']}")


if __name__ == "__main__":
    run_all_tests()