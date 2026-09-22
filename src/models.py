"""
src/models.py
Entrenamiento, evaluación comparativa y extracción de métricas de Machine Learning.
"""
import pandas as pd
from sklearn.dummy import DummyClassifier
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from src.config import PROCESSED_DATA_PATH, RANDOM_STATE, TEST_SIZE

FEATURE_COLS = ["calls_count", "minutes_spent", "messages_sent", "gb_used"]
TARGET_COL = "target_plan"


def get_data_splits(df: pd.DataFrame):
    """Genera las particiones estratificadas de entrenamiento y prueba."""
    X = df[FEATURE_COLS]
    y = df[TARGET_COL]
    return train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )


def build_models():
    """Diccionario con los estimadores evaluados en el proyecto."""
    return {
        "Baseline (Dummy)": DummyClassifier(strategy="most_frequent"),
        "Regresión Logística (Balanced)": Pipeline(
            [
                ("scaler", StandardScaler()),
                ("clf", LogisticRegression(class_weight="balanced", random_state=RANDOM_STATE)),
            ]
        ),
        "Random Forest (Optimizado Accuracy)": RandomForestClassifier(
            n_estimators=100, max_depth=7, min_samples_leaf=3, random_state=RANDOM_STATE
        ),
        "Random Forest (Balanced)": RandomForestClassifier(
            n_estimators=100, max_depth=6, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Gradient Boosting": GradientBoostingClassifier(
            n_estimators=100, max_depth=4, random_state=RANDOM_STATE
        ),
    }


def evaluate_models(X_train, X_test, y_train, y_test):
    """Entrena y evalúa todos los modelos calculando Accuracy, F1 y ROC-AUC."""
    models = build_models()
    records = []
    fitted_models = {}

    for name, model in models.items():
        model.fit(X_train, y_train)
        fitted_models[name] = model
        y_pred = model.predict(X_test)

        acc = accuracy_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred, zero_division=0)

        if hasattr(model, "predict_proba"):
            try:
                y_proba = model.predict_proba(X_test)[:, -1]
                roc = roc_auc_score(y_test, y_proba)
            except Exception:
                roc = 0.5
        else:
            roc = 0.5

        records.append({"Modelo": name, "Accuracy": acc, "F1-Score": f1, "ROC-AUC": roc})

    df_results = pd.DataFrame(records).sort_values(by="Accuracy", ascending=False)
    return df_results, fitted_models


def run_training():
    """Ejecución en consola del pipeline de modelos."""
    df = pd.read_parquet(PROCESSED_DATA_PATH)
    X_train, X_test, y_train, y_test = get_data_splits(df)

    results_table, fitted_models = evaluate_models(X_train, X_test, y_train, y_test)
    print("\n--- Evaluación Comparativa de Modelos ---")
    print(results_table.round(4).to_string(index=False))

    # Diagnóstico del mejor modelo en Accuracy
    best_rf = fitted_models["Random Forest (Optimizado Accuracy)"]
    preds = best_rf.predict(X_test)
    print("\n--- Reporte de Clasificación (Random Forest Optimizado) ---")
    print(classification_report(y_test, preds, target_names=["Surf", "Ultimate"]))


if __name__ == "__main__":
    run_training()