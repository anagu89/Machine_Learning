# src/generate_kpis.py
from __future__ import annotations
from pathlib import Path
import numpy as np
import pandas as pd

# --- Rutas del proyecto ---
BASE_DIR = Path(__file__).resolve().parents[1]
PROC_DIR = BASE_DIR / "data" / "processed"
ARTI_DIR = BASE_DIR / "docs" / "artifacts"
ARTI_DIR.mkdir(parents=True, exist_ok=True)

# --- Helpers ---
def ensure_dt(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    d["datum"] = pd.to_datetime(d["datum"], errors="coerce")
    return d.sort_values("datum").reset_index(drop=True)

def cov(series: pd.Series) -> float:
    s = pd.to_numeric(series, errors="coerce")
    m = s.mean()
    return (s.std(ddof=1) / m) if m and not np.isclose(m, 0.0) else np.nan

# --- Carga de datasets procesados (si existen) ---
def _maybe_read(path: Path) -> pd.DataFrame | None:
    try:
        return pd.read_csv(path) if path.exists() else None
    except Exception:
        return None

datasets = {}
m = _maybe_read(PROC_DIR / "saleshourly_clean.csv")
if m is not None: datasets["Hourly"]  = m
m = _maybe_read(PROC_DIR / "salesdaily_clean.csv")
if m is not None: datasets["Daily"]   = m
m = _maybe_read(PROC_DIR / "salesweekly_clean.csv")
if m is not None: datasets["Weekly"]  = m
m = _maybe_read(PROC_DIR / "salesmonthly_clean.csv")
if m is not None: datasets["Monthly"] = m

if not datasets:
    raise SystemExit("No se encontraron CSV en data/processed/*.csv. Ejecuta antes tu pipeline de limpieza.")

# --- KPIs  ---
kpis = []
for name, df in datasets.items():
    d = ensure_dt(df)
    s = pd.to_numeric(d["Total sales"], errors="coerce")

    kpis.append({
        "Dataset": name,
        "Filas": len(d),
        "Fecha mín": d["datum"].min(),
        "Fecha máx": d["datum"].max(),
        "Total ventas": s.sum(),
        "Media": s.mean(),
        "Mediana": s.median(),
        "Desv.Est.": s.std(ddof=1),
        "CoV": cov(s),
        "Mín": s.min(),
        "Máx": s.max(),
    })

kpis_df = pd.DataFrame(kpis).sort_values("Dataset")

# --- Renombrado para Streamlit (para la app.py) ---
kpis_app = kpis_df.rename(columns={
    "Dataset":   "dataset",
    "Filas":     "n_filas",
    "Fecha mín": "fecha_min",
    "Fecha máx": "fecha_max",
    "Media":     "media_total_sales",
})

# Guardamos el CSV final que leerá Streamlit
out_path = ARTI_DIR / "kpis_por_dataset.csv"
kpis_app.to_csv(out_path, index=False, encoding="utf-8")
print(f"[OK] KPIs exportados: {out_path}")