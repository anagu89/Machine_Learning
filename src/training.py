# src/training.py
"""
Crea y guarda los conjuntos train/test a partir de data/processed/*_clean.csv
Split temporal (por 'datum'): el 20% final se usa como test.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, Any

import numpy as np
import pandas as pd

# Fuerza backend no interactivo para scripts
import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns

# --- Modelado clásico / métricas
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.model_selection import TimeSeriesSplit, RandomizedSearchCV, GridSearchCV
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, mean_absolute_percentage_error, r2_score
)
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# SARIMAX
from statsmodels.tsa.statespace.sarimax import SARIMAX

# XGBoost (opcional)
try:
    from xgboost import XGBRegressor
except Exception:
    XGBRegressor = None

# CatBoost (opcional)
try:
    from catboost import CatBoostRegressor
except Exception:
    CatBoostRegressor = None

# --- Utils propios
from utils import (
    make_supervised_daily,
    train_test_last_fraction,
    evaluar,
    add_time_features_daily,   # <-- necesario para clustering
)

# ==========================
# Rutas
# ==========================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data" / "processed"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
ARTI_DIR = DOCS_DIR / "artifacts"
PLOTS_DIR = DOCS_DIR / "plots"

INPUT_DAILY   = DATA_DIR / "salesdaily_clean.csv"
INPUT_WEEKLY  = DATA_DIR / "salesweekly_clean.csv"
INPUT_MONTHLY = DATA_DIR / "salesmonthly_clean.csv"
INPUT_HOURLY  = DATA_DIR / "saleshourly_clean.csv"

# ==========================
# Helpers locales
# ==========================
def _safe_read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"No existe el fichero requerido: {path}")
    return pd.read_csv(path)

def _ensure_dirs():
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    ARTI_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

# === Helpers de FE/genéricos para otras granularidades ===
def _ensure_datetime_sorted(df, date_col="datum"):
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col], errors="coerce")
    d = d.dropna(subset=[date_col]).sort_values(date_col).reset_index(drop=True)
    return d

def _make_supervised_generic(df: pd.DataFrame, level: str):
    """
    Devuelve (X, y, fechas, feat_cols) con FE simple por granularidad:
      - Daily: usa make_supervised_daily(df) de utils
      - Weekly: Year, Week, lag_1, lag_2, roll4
      - Monthly: Year, Month, lag_1, lag_12, roll3
      - Hourly: Year, Month, Day, Hour, Weekday, lag_1, lag_24, roll24
    """
    lvl = (level or "").strip().lower()
    if lvl == "daily":
        return make_supervised_daily(df)

    d = _ensure_datetime_sorted(df, "datum").copy()

    # Asegurar Total sales (si no viene ya en el processed)
    if "Total sales" not in d.columns:
        atc_cols = [c for c in ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"] if c in d.columns]
        if atc_cols:
            d["Total sales"] = d[atc_cols].sum(axis=1, skipna=True)
        else:
            raise ValueError(f"{level}: falta 'Total sales' y no hay ATC para construirla.")

    # Partes comunes
    d["Year"] = d["datum"].dt.year
    d["Month"] = d["datum"].dt.month
    d["Weekday"] = d["datum"].dt.dayofweek + 1

    if lvl == "weekly":
        d["Week"] = d["datum"].dt.isocalendar().week.astype(int)
        d["lag_1"] = d["Total sales"].shift(1)
        d["lag_2"] = d["Total sales"].shift(2)
        d["roll4"] = d["Total sales"].shift(1).rolling(4, min_periods=2).mean()
        feat_cols = ["Year", "Week", "lag_1", "lag_2", "roll4"]

    elif lvl == "monthly":
        d["lag_1"]  = d["Total sales"].shift(1)
        d["lag_12"] = d["Total sales"].shift(12)
        d["roll3"]  = d["Total sales"].shift(1).rolling(3, min_periods=2).mean()
        feat_cols = ["Year", "Month", "lag_1", "lag_12", "roll3"]

    elif lvl == "hourly":
        d["Day"]  = d["datum"].dt.day
        d["Hour"] = d["datum"].dt.hour
        d["lag_1"]  = d["Total sales"].shift(1)
        d["lag_24"] = d["Total sales"].shift(24)
        d["roll24"] = d["Total sales"].shift(1).rolling(24, min_periods=6).mean()
        feat_cols = ["Year", "Month", "Day", "Hour", "Weekday", "lag_1", "lag_24", "roll24"]

    else:
        raise ValueError(f"Granularidad no soportada: {level}")

    d = d.dropna(subset=feat_cols + ["Total sales"]).reset_index(drop=True)
    X = d[feat_cols].astype(float).copy()
    y = d["Total sales"].astype(float).values
    fechas = pd.to_datetime(d["datum"]).values
    return X, y, fechas, feat_cols

def _save_split(X, y, fechas, feat_cols, out_train: Path, out_test: Path, frac=0.2):
    n = len(X)
    cut = int(n*(1-frac))
    X_train, X_test = X[:cut], X[cut:]
    y_train, y_test = y[:cut], y[cut:]
    fechas_train, fechas_test = fechas[:cut], fechas[cut:]

    pd.DataFrame(X_train, columns=feat_cols) \
      .assign(y=y_train, fecha=fechas_train) \
      .to_csv(out_train, index=False)

    pd.DataFrame(X_test, columns=feat_cols) \
      .assign(y=y_test, fecha=fechas_test) \
      .to_csv(out_test, index=False)

    print(f"[OK] Guardado: {out_train}")
    print(f"[OK] Guardado: {out_test}")

# ==== Guardado de modelos (lo subimos aquí para poder usarlo en KMeans también)
import joblib as _joblib

MODEL_ORDER = {
    "Prophet": 1,
    "ARIMA": 2,
    "RandomForest": 3,
    "GradientBoosting": 4,
    "XGBoost": 5,
    "CatBoost": 6,
    "Clustering": 7,
}

def save_model(model, label: str):
    num = None
    for k, v in MODEL_ORDER.items():
        if k.lower() in label.lower():
            num = v
            break
    if num is None:
        num = 99
    fname = f"trained_model_{num}_{label}.pkl"
    out_path = MODELS_DIR / fname
    _joblib.dump(model, out_path)
    print(f"[OK] Guardado modelo: {out_path.relative_to(BASE_DIR)}")

# === Persistencia de importancias de variables ===
def _save_feature_importances(pipe, model_key: str, feat_names):
    try:
        model = pipe.named_steps.get("model", None)
        if model is None:
            return
        if hasattr(model, "feature_importances_"):
            importancias = pd.Series(model.feature_importances_, index=feat_names).sort_values(ascending=False)
            out_csv = ARTI_DIR / f"feature_importances_{model_key}.csv"
            importancias.to_csv(out_csv, header=["importance"])
            print(f"[OK] Guardado: {out_csv}")

            top = importancias.head(15).iloc[::-1]
            plt.figure(figsize=(8, max(3, 0.4*len(top))))
            plt.barh(top.index, top.values)
            plt.title(f"Feature importances — {model_key}")
            plt.tight_layout()
            plt.savefig(PLOTS_DIR / f"feature_importances_{model_key}.png", dpi=150)
            plt.close()
    except Exception as e:
        print(f"[AVISO] No se pudieron guardar importancias de {model_key}: {e}")

# ==========================
# Entrenamiento
# ==========================
def train_and_evaluate() -> Dict[str, Any]:
    _ensure_dirs()

    # Cargar datos (principalmente Daily)
    daily = _safe_read_csv(INPUT_DAILY).copy()
    weekly = _safe_read_csv(INPUT_WEEKLY).copy()
    monthly = _safe_read_csv(INPUT_MONTHLY).copy()
    hourly = _safe_read_csv(INPUT_HOURLY).copy()

    # --- 1) Prophet

    # --- 2) SARIMAX baseline (Daily)
    d = daily.copy()
    d["datum"] = pd.to_datetime(d["datum"], errors="coerce")
    d = d.dropna(subset=["datum", "Total sales"]).sort_values("datum")
    y_all = pd.to_numeric(d["Total sales"], errors="coerce").fillna(0.0).values
    cut = int(len(y_all) * 0.8)
    y_tr, y_te = y_all[:cut], y_all[cut:]

    sarimax = SARIMAX(
        y_tr,
        order=(1, 1, 1),
        seasonal_order=(1, 0, 1, 7),
        enforce_stationarity=False,
        enforce_invertibility=False
    ).fit(disp=False)
    yhat = sarimax.forecast(steps=len(y_te))
    res_sarimax = evaluar(y_te, yhat)

    # Preparar datos supervisados
    X, y, fechas, feat_cols = make_supervised_daily(daily)
    X_train, X_test, y_train, y_test = train_test_last_fraction(X, y, frac=0.2)

    # =========================
    # Guardar datasets train/test en CSV (Daily/Weekly/Monthly/Hourly)
    # =========================
    TRAIN_DIR = BASE_DIR / "data" / "train"
    TEST_DIR  = BASE_DIR / "data" / "test"
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    TEST_DIR.mkdir(parents=True, exist_ok=True)

    inputs = {
        "Daily":   INPUT_DAILY,
        "Weekly":  INPUT_WEEKLY,
        "Monthly": INPUT_MONTHLY,
        "Hourly":  INPUT_HOURLY,
    }

    for level, path in inputs.items():
        if not path.exists():
            print(f"[AVISO] No existe {path}. Omito {level}.")
            continue

        df_src = _safe_read_csv(path)
        if level == "Daily":
            Xs, ys, fechas_s, feat_cols_s = make_supervised_daily(df_src)
        else:
            Xs, ys, fechas_s, feat_cols_s = _make_supervised_generic(df_src, level)

        out_train = TRAIN_DIR / f"{level.lower()}_train.csv"
        out_test  = TEST_DIR  / f"{level.lower()}_test.csv"
        _save_split(Xs, ys, fechas_s, feat_cols_s, out_train, out_test, frac=0.2)

    print(f"[OK] Guardados splits en {TRAIN_DIR} y {TEST_DIR}")

    # --- 3) Random Forest
    rf_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", RandomForestRegressor(
            n_estimators=300, max_depth=None, min_samples_split=2,
            random_state=42, n_jobs=-1
        ))
    ])
    rf_pipe.fit(X_train, y_train)
    rf_base = evaluar(y_test, rf_pipe.predict(X_test))

    tscv = TimeSeriesSplit(n_splits=3)
    rf_search = RandomizedSearchCV(
        estimator=rf_pipe,
        param_distributions={
            "model__n_estimators": [200, 300, 500, 800],
            "model__max_depth": [None, 6, 10, 14],
            "model__min_samples_split": [2, 5, 10],
            "model__min_samples_leaf": [1, 2, 4],
            "model__max_features": ["sqrt", "log2", None]
        },
        n_iter=12, cv=tscv, scoring="neg_mean_absolute_error",
        random_state=42, n_jobs=-1, refit=True
    )
    rf_search.fit(X_train, y_train)
    rf_best = rf_search.best_estimator_
    rf_best_metrics = evaluar(y_test, rf_best.predict(X_test))

    # --- 4) Gradient Boosting
    gbr_pipe = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("model", GradientBoostingRegressor(
            n_estimators=400, learning_rate=0.05, max_depth=3, random_state=42
        ))
    ])
    gbr_pipe.fit(X_train, y_train)
    gbr_base = evaluar(y_test, gbr_pipe.predict(X_test))

    gbr_grid = GridSearchCV(
        estimator=gbr_pipe,
        param_grid={
            "model__n_estimators": [200, 400, 600],
            "model__learning_rate": [0.03, 0.05, 0.1],
            "model__max_depth": [2, 3, 4],
        },
        cv=3, n_jobs=-1, scoring="neg_mean_absolute_error", refit=True
    )
    gbr_grid.fit(X_train, y_train)
    gbr_best = gbr_grid.best_estimator_
    gbr_best_metrics = evaluar(y_test, gbr_best.predict(X_test))

    # --- 5) XGBoost (opcional)
    xgb_pipe = None; xgb_best = None; xgb_base = None; xgb_best_metrics = None
    if XGBRegressor is not None:
        xgb_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", XGBRegressor(
                n_estimators=600, max_depth=4, learning_rate=0.05,
                subsample=0.8, colsample_bytree=0.8, reg_lambda=1.0,
                random_state=42, n_jobs=-1
            ))
        ])
        xgb_pipe.fit(X_train, y_train)
        xgb_base = evaluar(y_test, xgb_pipe.predict(X_test))

        xgb_search = RandomizedSearchCV(
            estimator=xgb_pipe,
            param_distributions={
                "model__n_estimators": [400, 600, 800],
                "model__max_depth": [3, 4, 5],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__colsample_bytree": [0.7, 0.8, 1.0],
                "model__reg_lambda": [0.5, 1.0, 2.0],
            },
            n_iter=12, cv=tscv, scoring="neg_mean_absolute_error",
            random_state=42, n_jobs=-1, refit=True
        )
        xgb_search.fit(X_train, y_train)
        xgb_best = xgb_search.best_estimator_
        xgb_best_metrics = evaluar(y_test, xgb_best.predict(X_test))
    else:
        print("[AVISO] XGBoost no disponible. `pip install xgboost` para activarlo.")

    # --- 6) CatBoost (opcional)
    cat_pipe = None; cat_best = None; cat_base = None; cat_best_metrics = None
    if CatBoostRegressor is not None:
        cat_pipe = Pipeline(steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("model", CatBoostRegressor(
                loss_function="RMSE",
                random_seed=42,
                logging_level="Silent",
            ))
        ])
        cat_pipe.fit(X_train, y_train)
        cat_base = evaluar(y_test, cat_pipe.predict(X_test))

        cat_search = RandomizedSearchCV(
            estimator=cat_pipe,
            param_distributions={
                "model__n_estimators": [400, 600, 800],
                "model__depth": [4, 6, 8],
                "model__learning_rate": [0.03, 0.05, 0.1],
                "model__subsample": [0.7, 0.8, 1.0],
                "model__l2_leaf_reg": [1.0, 3.0, 5.0, 7.0],
            },
            n_iter=12, cv=tscv, n_jobs=-1, random_state=42,
            scoring="neg_mean_absolute_error", refit=True
        )
        cat_search.fit(X_train, y_train)
        cat_best = cat_search.best_estimator_
        cat_best_metrics = evaluar(y_test, cat_best.predict(X_test))
    else:
        print("[AVISO] CatBoost no disponible. `pip install catboost` para activarlo.")

    # --- 7) KMeans Clustering (Daily)
    d_feats, feat_cols_clust = add_time_features_daily(daily)
    d_feats = d_feats.dropna(subset=["Total sales"]).copy()

    cols_seg = [c for c in ["Total sales", "lag_1", "lag_7", "roll7", "Month", "Weekday"] if c in d_feats.columns]
    Z = d_feats[cols_seg].astype(float)

    best_k, best_score, best_pipe = None, -1.0, None
    for k in [2, 3, 4, 5]:
        pipe = Pipeline(steps=[
            ("scaler", StandardScaler()),
            ("kmeans", KMeans(n_clusters=k, n_init="auto", random_state=42))
        ])
        pipe.fit(Z)
        Zs = pipe.named_steps["scaler"].transform(Z)
        labels = pipe.named_steps["kmeans"].labels_
        s = silhouette_score(Zs, labels)
        if s > best_score:
            best_k, best_score, best_pipe = k, s, pipe

    labels = best_pipe.named_steps["kmeans"].labels_
    d_clu = d_feats.copy()
    d_clu["cluster"] = labels

    resumen = d_clu.groupby("cluster").agg(
        dias=("cluster","count"),
        venta_media=("Total sales","mean"),
        venta_mediana=("Total sales","median"),
        venta_p95=("Total sales", lambda x: np.percentile(x,95))
    ).sort_values("venta_media", ascending=False)

    print(f"\n=== KMeans (Pipeline) ===\nMejor k={best_k} | silhouette={best_score:.3f}")
    print("\nResumen clusters (días):")
    print(resumen.round(3).to_string())
    print("\n=== Métrica de clustering (no supervisado) ===")
    print(pd.DataFrame([{"Silhouette": best_score, "k": best_k}], index=["KMeans"]).round(3).to_string())

    # Artefactos clustering
    d_out = d_clu[["datum", "Total sales", "cluster"] + cols_seg].copy()
    d_out.rename(columns={"datum": "fecha", "Total sales": "ventas"}, inplace=True)
    d_out.to_csv(ARTI_DIR / "daily_clusters.csv", index=False)
    resumen.round(4).to_csv(ARTI_DIR / "cluster_summary.csv")
    pd.DataFrame([{"Silhouette": best_score, "k": best_k}], index=["KMeans"]).to_csv(ARTI_DIR / "cluster_metric.csv")

    pca = PCA(n_components=2, random_state=42)
    Zs = best_pipe.named_steps["scaler"].transform(Z)
    XY = pca.fit_transform(Zs)

    plt.figure(figsize=(8,6))
    plt.scatter(XY[:,0], XY[:,1], c=labels, s=12)
    plt.title(f"KMeans PCA scatter — k={best_k} (silhouette={best_score:.3f})")
    plt.xlabel("PC1"); plt.ylabel("PC2")
    plt.grid(alpha=0.3); plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_pca_scatter.png", dpi=150); plt.close()

    heat_df = d_clu.groupby("cluster")[cols_seg].mean().round(3)
    plt.figure(figsize=(10, max(3, 0.5*len(cols_seg))))
    sns.heatmap(heat_df, annot=True, fmt=".2f", cmap="viridis")
    plt.title("Medias por cluster (features de segmentación)")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_features_heatmap.png", dpi=150); plt.close()

    sizes = d_clu["cluster"].value_counts().sort_index()
    plt.figure(figsize=(6,6))
    plt.pie(sizes, labels=[f"C{c}" for c in sizes.index], autopct="%1.1f%%", startangle=90)
    plt.title("Distribución de tamaños de cluster")
    plt.tight_layout()
    plt.savefig(PLOTS_DIR / "kmeans_cluster_sizes.png", dpi=150); plt.close()

    # Guardar modelo de clustering
    km_label = f"Clustering_KMeans_k{best_k}"
    save_model(best_pipe, km_label)

    # ==========================
    # Comparativa supervisados
    # ==========================
    resultados: Dict[str, Dict[str, float]] = {}
    resultados["SARIMAX"] = res_sarimax
    resultados["RandomForest (base)"] = rf_base
    resultados["RandomForest (best)"] = rf_best_metrics
    resultados["GradientBoosting (base)"] = gbr_base
    resultados["GradientBoosting (best)"] = gbr_best_metrics
    if xgb_base is not None:
        resultados["XGBoost (base)"] = xgb_base
    if xgb_best_metrics is not None:
        resultados["XGBoost (best)"] = xgb_best_metrics
    if cat_base is not None:
        resultados["CatBoost (base)"] = cat_base
    if cat_best_metrics is not None:
        resultados["CatBoost (best)"] = cat_best_metrics

    df_resumen = pd.DataFrame(resultados).T[["MAPE", "MAE", "MSE", "R2"]].sort_values("MAPE")
    print("\n=== Comparativa de modelos ===")
    print(df_resumen.round(3).to_string())

    # Guardar importancias (si existen)
    _save_feature_importances(rf_best,  "RF_best",  feat_cols)
    _save_feature_importances(gbr_best, "GBR_best", feat_cols)
    if xgb_best is not None:
        _save_feature_importances(xgb_best, "XGB_best", feat_cols)
    if cat_best is not None:
        _save_feature_importances(cat_best, "CAT_best", feat_cols)

    # Gráfico comparativa MAPE (bloque único)
    try:
        df_plot = df_resumen.sort_values("MAPE")
        plt.figure(figsize=(10, max(3, 0.6*len(df_plot))))
        plt.barh(df_plot.index, df_plot["MAPE"])
        plt.xlabel("MAPE (%)")
        plt.title("Comparativa de modelos (MAPE menor es mejor)")
        plt.tight_layout()
        plt.savefig(PLOTS_DIR / "model_comparison_mape.png", dpi=150)
        plt.close()
        print(f"[OK] Figura guardada: {PLOTS_DIR / 'model_comparison_mape.png'}")
    except Exception as e:
        print(f"[AVISO] No se pudo guardar la comparativa MAPE: {e}")

    # ==========================
    # Guardar modelos supervisados
    # ==========================
    modelos_entrenados = {
        "RandomForest_base": rf_pipe,
        "RandomForest_best": rf_best,
        "GradientBoosting_base": gbr_pipe,
        "GradientBoosting_best": gbr_best,
    }
    if xgb_pipe is not None:
        modelos_entrenados["XGBoost_base"] = xgb_pipe
    if xgb_best is not None:
        modelos_entrenados["XGBoost_best"] = xgb_best
    if cat_pipe is not None:
        modelos_entrenados["CatBoost_base"] = cat_pipe
    if cat_best is not None:
        modelos_entrenados["CatBoost_best"] = cat_best

    for label, modelo in modelos_entrenados.items():
        save_model(modelo, label)

    # ==========================
    # Elegir modelo final (mínima MAPE)
    # ==========================
    top_name = df_resumen.index[0]
    name_map = {
        "RandomForest (best)": rf_best,
        "RandomForest (base)": rf_pipe,
        "GradientBoosting (best)": gbr_best,
        "GradientBoosting (base)": gbr_pipe,
        "XGBoost (best)": xgb_best,
        "XGBoost (base)": xgb_pipe,
        "CatBoost (best)": cat_best,
        "CatBoost (base)": cat_pipe,
        "SARIMAX": sarimax,
    }
    final_model = name_map.get(top_name, rf_best)

    print(f"[OK] Modelo final elegido: {top_name}")

    import pickle
    with open(MODELS_DIR / "final_model.pkl", "wb") as f:
        pickle.dump(final_model, f)
    print(f"[OK] Modelo final guardado en { (MODELS_DIR / 'final_model.pkl').relative_to(BASE_DIR) }")

    # ==========================
    # Exportar métricas
    # ==========================
    out_csv = ARTI_DIR / "metrics_summary.csv"
    df_resumen.round(3).to_csv(out_csv, index=True)
    print(f"[OK] Guardado: {out_csv.relative_to(BASE_DIR)}")

    md_path = ARTI_DIR / "metrics_summary.md"
    try:
        md_text = df_resumen.round(3).to_markdown(index=True)
    except Exception:
        md_text = "```\n" + df_resumen.round(3).to_string() + "\n```"
    with open(md_path, "w", encoding="utf-8") as f:
        f.write(md_text)
    print(f"[OK] Guardado: {md_path.relative_to(BASE_DIR)}")

    return {
        "df_resumen": df_resumen,
        "feats": feat_cols,
    }

# ==========================
# Main
# ==========================
if __name__ == "__main__":
    try:
        train_and_evaluate()
    except Exception as e:
        print(f"[ERROR] training falló: {e}")
        sys.exit(1)