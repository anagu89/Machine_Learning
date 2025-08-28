# src/evaluation.py
"""
Evaluación del modelo final:
 - Carga models/final_model.pkl
 - Carga data/test/daily_test.csv (o reconstruye test desde data/processed/salesdaily_clean.csv)
 - Repite feature engineering de entrenamiento (Daily)
 - Predice, calcula métricas y guarda artefactos y figura

Ejecución (desde la raíz del proyecto):
    python src/evaluation.py
"""

from __future__ import annotations

import os
import sys
import pickle
from pathlib import Path
from datetime import datetime

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# --- Rutas base ---
BASE_DIR = Path(__file__).resolve().parents[1]
MODELS_DIR = BASE_DIR / "models"
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"

DOCS_DIR = BASE_DIR / "docs"
ARTI_DIR = DOCS_DIR / "artifacts"
PLOTS_DIR = DOCS_DIR / "plots"

ARTEFACTS_TO_CREATE = [ARTI_DIR, PLOTS_DIR]
for d in ARTEFACTS_TO_CREATE:
    d.mkdir(parents=True, exist_ok=True)

# --- Configuración ATC (por consistencia con notebooks) ---
ATC_COLS = ['M01AB','M01AE','N02BA','N02BE','N05B','N05C','R03','R06']


# =========================
# Utilidades métricas/FE
# =========================
def mape(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    mask = y_true != 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100)

def evaluar(y_true, y_pred):
    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
    return {
        "MAPE": mape(y_true, y_pred),
        "MAE": float(mean_absolute_error(y_true, y_pred)),
        "MSE": float(mean_squared_error(y_true, y_pred)),
        "R2":  float(r2_score(y_true, y_pred)),
    }

def ensure_datetime_sorted(df, date_col="datum"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return d

def ensure_total_sales(df):
    d = df.copy()
    if "Total sales" not in d.columns:
        cols_present = [c for c in ATC_COLS if c in d.columns]
        if cols_present:
            d["Total sales"] = d[cols_present].sum(axis=1, skipna=True)
        else:
            raise ValueError("No existe 'Total sales' ni columnas ATC para construirla.")
    return d

def add_time_features_daily(df):
    """
    Misma lógica que en el notebook:
      - Year, Month, Weekday
      - lag_1, lag_7
      - roll7 (media móvil de 7 con lag 1)
      - + ATC presentes (si existen)
    """
    d = ensure_datetime_sorted(df, "datum")
    d = ensure_total_sales(d)

    d["Year"] = d["datum"].dt.year
    d["Month"] = d["datum"].dt.month
    d["Weekday"] = d["datum"].dt.dayofweek + 1

    d["lag_1"] = d["Total sales"].shift(1)
    d["lag_7"] = d["Total sales"].shift(7)
    d["roll7"] = d["Total sales"].shift(1).rolling(7, min_periods=3).mean()

    d = d.dropna().reset_index(drop=True)

    feat_cols = ["Year","Month","Weekday","lag_1","lag_7","roll7"]
    atc_present = [c for c in ATC_COLS if c in d.columns]
    feat_cols = feat_cols + atc_present

    X = d[feat_cols].astype(float).copy()
    y = d["Total sales"].astype(float).values
    fechas = pd.to_datetime(d["datum"]).values
    return X, y, fechas, feat_cols


def align_features_to_model(X, model):
    """
    Reordena/ajusta X a las columnas que espera el modelo (feature_names_in_ si existe).
    Si faltan columnas, se crean con 0; si sobran, se descartan.
    """
    if hasattr(model, "feature_names_in_"):
        expected = list(model.feature_names_in_)
        X_aligned = X.copy()
        # añadir faltantes
        for col in expected:
            if col not in X_aligned.columns:
                X_aligned[col] = 0.0
        # quedarnos solo con expected y en el mismo orden
        X_aligned = X_aligned[expected]
        return X_aligned
    else:
        # No hay info del modelo, usamos las columnas de X tal cual
        return X


def save_markdown_table(df, path):
    try:
        md = df.to_markdown(index=True)
    except Exception:
        md = "```\n" + df.to_string() + "\n```"
    with open(path, "w", encoding="utf-8") as f:
        f.write(md)


# =========================
# Carga del modelo final
# =========================
def load_final_model(path: Path):
    if not path.exists():
        raise FileNotFoundError(f"No se encontró el modelo final en: {path}")
    # se guardó con pickle en training, intentamos pickle primero
    try:
        with open(path, "rb") as f:
            return pickle.load(f)
    except Exception:
        # fallback a joblib si fuese el caso
        import joblib
        return joblib.load(path)


# =========================
# Carga del dataset de test
# =========================
def load_test_daily() -> pd.DataFrame:
    """
    Prioridad:
      1) data/test/daily_test.csv
      2) (fallback) data/processed/salesdaily_clean.csv (tomando último 20% tras FE)
    """
    candidate = TEST_DIR / "daily_test.csv"
    if candidate.exists():
        print(f"[INFO] Cargando test desde: {candidate.relative_to(BASE_DIR)}")
        return pd.read_csv(candidate)

    # Fallback
    fallback = PROC_DIR / "salesdaily_clean.csv"
    if not fallback.exists():
        raise FileNotFoundError(
            "No se encontró data/test/daily_test.csv ni data/processed/salesdaily_clean.csv"
        )
    print(f"[AVISO] No existe data/test/daily_test.csv. Usaré tramo de test del processed: {fallback.relative_to(BASE_DIR)}")
    return pd.read_csv(fallback)


# =========================
# Gráfica final
# =========================
def plot_and_save(y_true, y_pred, fechas, title_prefix="Predicción vs Realidad", modelo="Modelo final"):
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    fechas = pd.to_datetime(fechas)

    # Alinear longitudes
    n = min(len(y_true), len(y_pred), len(fechas))
    y_true = y_true[-n:]
    y_pred = y_pred[-n:]
    fechas = fechas[-n:]

    # banda de error
    abs_err = np.abs(y_true - y_pred)

    plt.figure(figsize=(12, 5))
    plt.plot(fechas, y_true, label="Real", lw=2)
    plt.plot(fechas, y_pred, label=f"Pred ({modelo})", lw=2)
    plt.fill_between(fechas, y_pred - abs_err, y_pred + abs_err, alpha=0.15, label="|Error|")
    res = evaluar(y_true, y_pred)
    plt.title(f"{title_prefix} — {modelo}  |  MAPE={res['MAPE']:.2f}%  MAE={res['MAE']:.2f}")
    plt.xlabel("Fecha"); plt.ylabel("Ventas")
    plt.grid(True, alpha=0.3); plt.legend()
    plt.tight_layout()

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_png = PLOTS_DIR / f"pred_vs_real_{ts}.png"
    plt.savefig(out_png, dpi=150)
    plt.close()
    print(f"[OK] Figura guardada: {out_png.relative_to(BASE_DIR)}")
    return out_png


# =========================
# Main
# =========================
def main():
    # 1) Modelo
    final_model_path = MODELS_DIR / "final_model.pkl"
    print(f"[INFO] Cargando modelo: {final_model_path.relative_to(BASE_DIR)}")
    model = load_final_model(final_model_path)

    # 2) Datos test
    df_test_raw = load_test_daily()

    # --- Detectamos formato del CSV de test ---
    #   A) Formato "crudo": tiene 'datum' (se generan features)
    #   B) Formato "features": no tiene 'datum' pero sí 'fecha', 'y' y columnas de entrada (como guardó training.py)

    cols_lower = {c.lower() for c in df_test_raw.columns}

    if (TEST_DIR / "daily_test.csv").exists():
        if "datum" in cols_lower:
            # ===== Caso A: test crudo (con 'datum') =====
            X_all, y_all, fechas_all, feat_cols = add_time_features_daily(df_test_raw)
            X_test, y_test, fechas_test = X_all, y_all, fechas_all
            print(f"[INFO] Test detectado 'crudo': {len(X_test)} filas tras FE")
        else:
            # ===== Caso B: test ya con features (como lo guarda training.py) =====
            print(f"[INFO] Test detectado con features: usando columnas existentes")

            # Columnas de fechas/target
            has_fecha = "fecha" in cols_lower
            has_y     = "y" in cols_lower

            # Columnas esperadas por el modelo (si las expone)
            if hasattr(model, "feature_names_in_"):
                feat_cols = list(model.feature_names_in_)
            else:
                # Fallback: todas menos 'y', 'fecha', 'datum'
                feat_cols = [c for c in df_test_raw.columns if c.lower() not in {"y", "fecha", "datum"}]

            # Construimos X_test (solo re-ordenaremos más adelante con align_features_to_model)
            X_test = df_test_raw[feat_cols].copy()

            # y_test (si está)
            y_test = df_test_raw["y"].values if has_y else None

            # Fechas (si están)
            if has_fecha:
                fechas_test = pd.to_datetime(df_test_raw["fecha"], errors="coerce").values
            else:
                fechas_test = np.arange(len(X_test))
    else:
        # ===== Fallback: usar processed completo y quedarnos con el último 20% como test =====
        X_all, y_all, fechas_all, feat_cols = add_time_features_daily(df_test_raw)
        cut = int(len(X_all) * 0.8)
        X_test, y_test, fechas_test = X_all[cut:], y_all[cut:], fechas_all[cut:]
        print(f"[INFO] Test reconstruido desde processed: {len(X_test)} filas (último 20%)")

    # 3) Alinear columnas con el modelo y predecir
    X_test_aligned = align_features_to_model(pd.DataFrame(X_test, columns=X_test.columns), model)
    y_pred = np.asarray(model.predict(X_test_aligned), dtype=float)

    # 4) Métricas (solo si tenemos y_test)
    if y_test is not None:
        metrics = evaluar(y_test, y_pred)
        metrics_df = pd.DataFrame([metrics], index=["final_model"])
        print("\n=== Métricas (test) ===")
        print(metrics_df.round(3).to_string())

        # Guardado de artefactos
        eval_csv = ARTI_DIR / "eval_metrics.csv"
        metrics_df.round(4).to_csv(eval_csv, index=True)
        print(f"[OK] Guardado: {eval_csv.relative_to(BASE_DIR)}")

        eval_md = ARTI_DIR / "eval_metrics.md"
        save_markdown_table(metrics_df.round(4), eval_md)
        print(f"[OK] Guardado: {eval_md.relative_to(BASE_DIR)}")
    else:
        print("\n[AVISO] El fichero de test no incluye la columna 'y'; se omiten métricas y solo se exportan predicciones.")

    # Exportar y_true / y_pred (si hay y) y fechas
    yp_dict = {"fecha": pd.to_datetime(fechas_test)}
    yp_dict["y_pred"] = y_pred
    if y_test is not None:
        yp_dict["y_true"] = y_test

    yp_csv = ARTI_DIR / "y_true_pred.csv"
    pd.DataFrame(yp_dict).to_csv(yp_csv, index=False)
    print(f"[OK] Guardado: {yp_csv.relative_to(BASE_DIR)}")

    # 5) Gráfica
    try:
        y_plot = y_test if y_test is not None else y_pred  # si no hay y, graficamos solo pred
        plot_and_save(y_plot, y_pred, fechas_test, modelo="Modelo final")
    except Exception as e:
        print(f"[AVISO] No se pudo guardar la figura: {e}")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[ERROR] evaluation.py falló: {e}")
        sys.exit(1)