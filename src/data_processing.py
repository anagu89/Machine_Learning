# src/data_processing.py
"""
Pipeline reproducible de limpieza:
  - Lee data/raw/*.csv
  - Limpia y enriquece (fechas, Total sales, partes temporales)
  - Escribe data/processed/*_clean.csv
  - Exporta artefactos de negocio (KPIs, ranking categorías, mix HHI)
  - Genera plots ACF/PACF con nombres alineados a los notebooks

Ejecución (desde la raíz del proyecto):
    python src/data_processing.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict

import pandas as pd
import numpy as np

# Forzar backend no interactivo para evitar problemas de Tkinter
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

from utils import (
    CleaningConfig,
    ensure_datetime,
    compute_total_sales,
    add_time_parts,
)

# -----------------------------
# Rutas de proyecto
# -----------------------------
BASE_DIR = Path(__file__).resolve().parents[1]
RAW_DIR = BASE_DIR / "data" / "raw"
PROC_DIR = BASE_DIR / "data" / "processed"

DOCS_DIR = BASE_DIR / "docs"
ARTI_DIR = DOCS_DIR / "artifacts"
PLOTS_DIR = DOCS_DIR / "plots"

INPUT_FILES = {
    "Hourly": RAW_DIR / "saleshourly.csv",
    "Daily": RAW_DIR / "salesdaily.csv",
    "Weekly": RAW_DIR / "salesweekly.csv",
    "Monthly": RAW_DIR / "salesmonthly.csv",
}

OUTPUT_FILES = {
    "Hourly": PROC_DIR / "saleshourly_clean.csv",
    "Daily": PROC_DIR / "salesdaily_clean.csv",
    "Weekly": PROC_DIR / "salesweekly_clean.csv",
    "Monthly": PROC_DIR / "salesmonthly_clean.csv",
}


# -----------------------------
# Funciones
# -----------------------------
def clean_single(df: pd.DataFrame, level: str, cfg: CleaningConfig) -> pd.DataFrame:
    """
    Limpieza mínima + enriquecimiento coherente entre notebooks y src/:
      1) Asegurar datetime y orden temporal
      2) Convertir ATC a numérico y crear 'Total sales'
      3) Añadir columnas temporales estándar (según granularidad)
      4) Reordenado ligero (opcional y seguro)
    """
    d = ensure_datetime(df, "datum")
    d = compute_total_sales(d, cfg.atc_cols)
    d = add_time_parts(d, level)

    # Reordenado opcional: poner 'datum' y 'Total sales' al principio si existen
    cols = list(d.columns)
    front = [c for c in ["datum", "Total sales"] if c in cols]
    rest = [c for c in cols if c not in front]
    d = d[front + rest]

    # Sin imputar ni eliminar outliers aquí: eso se deja para modelado
    return d


def run_processing() -> Dict[str, pd.DataFrame]:
    """Carga → limpia → guarda. Devuelve dict con DataFrames limpios."""
    cfg = CleaningConfig()
    PROC_DIR.mkdir(parents=True, exist_ok=True)
    ARTI_DIR.mkdir(parents=True, exist_ok=True)
    PLOTS_DIR.mkdir(parents=True, exist_ok=True)

    cleaned: Dict[str, pd.DataFrame] = {}

    for level, in_path in INPUT_FILES.items():
        if not in_path.exists():
            print(f"[AVISO] No existe: {in_path}")
            continue

        print(f"[INFO] Cargando: {in_path.name}")
        df = pd.read_csv(in_path)

        d_clean = clean_single(df, level=level, cfg=cfg)
        out_path = OUTPUT_FILES[level]
        d_clean.to_csv(out_path, index=False)
        print(f"[OK] Guardado limpio: {out_path.relative_to(BASE_DIR)}  (filas={len(d_clean)}, cols={len(d_clean.columns)})")

        cleaned[level] = d_clean

    print("\n[FIN] Limpieza completada. Ficheros en data/processed/:")
    for level, out_path in OUTPUT_FILES.items():
        if out_path.exists():
            print(f"  - {out_path.relative_to(BASE_DIR)}")

    # -----------------------------
    # Artefactos de negocio
    # -----------------------------
    try:
        # KPIs por dataset (filas, rango fechas, media ventas)
        rows = []
        for level, path in OUTPUT_FILES.items():
            if not path.exists():
                continue
            d = pd.read_csv(path)
            d["datum"] = pd.to_datetime(d["datum"], errors="coerce")
            d = d.dropna(subset=["datum"])
            mean_sales = pd.to_numeric(d.get("Total sales", pd.Series(dtype=float)), errors="coerce").mean()
            rows.append({
                "dataset": level,
                "n_filas": len(d),
                "fecha_min": d["datum"].min(),
                "fecha_max": d["datum"].max(),
                "media_total_sales": mean_sales,
            })
        if rows:
            kpis_df = pd.DataFrame(rows)
            kpis_path = ARTI_DIR / "kpis_por_dataset.csv"
            kpis_df.to_csv(kpis_path, index=False)
            print(f"[OK] Guardado: {kpis_path.relative_to(BASE_DIR)}")

        # Ranking de categorías (Daily)
        daily_path = OUTPUT_FILES.get("Daily")
        if daily_path and daily_path.exists():
            dd = pd.read_csv(daily_path)
            drugs = [c for c in ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"] if c in dd.columns]
            if drugs:
                category_totals = dd[drugs].sum().sort_values(ascending=False)
                rank_path = ARTI_DIR / "ranking_categorias_daily.csv"
                category_totals.to_frame("ventas").to_csv(rank_path)
                print(f"[OK] Guardado: {rank_path.relative_to(BASE_DIR)}")

                # Mix categorías + HHI
                total = category_totals.sum()
                share = (category_totals / total).fillna(0.0)
                hhi = float((share**2).sum())
                mix_df = pd.DataFrame({
                    "categoria": category_totals.index,
                    "ventas": category_totals.values,
                    "share": share.values
                })
                mix_path = ARTI_DIR / "mix_categorias_hhi.csv"
                mix_df.to_csv(mix_path, index=False)
                print(f"[OK] Guardado: {mix_path.relative_to(BASE_DIR)}")
                # (Se informa HHI por consola)
                print(f"[INFO] HHI (Daily, mix categorías) = {hhi:.4f}")
    except Exception as e:
        print(f"[AVISO] No se pudieron generar artefactos de negocio: {e}")

    # -----------------------------
    # ACF / PACF (EDA) con nombres fijos
    # -----------------------------
    def _safe_nlags(n: int, prefer: int = 40) -> int:
        """Devuelve un nlags seguro para ACF/PACF (PACF exige nlags < n/2)."""
        return max(1, min(prefer, (n // 2) - 1))

    def _save_acf_pacf_named(series: np.ndarray, title_prefix: str, acf_path: Path, pacf_path: Path, prefer_lags: int = 40):
        """
        Guarda ACF y PACF con nombres EXACTOS (sin añadir sufijos automáticos).
        - series: array 1D sin NaNs
        - acf_path / pacf_path: rutas completas al PNG final
        """
        series = np.asarray(series, dtype=float)
        n = len(series)

        if n < 10 or np.allclose(series, series[0]):
            print(f"[AVISO] Serie demasiado corta/constante para ACF/PACF: n={n}. Se omite.")
            return

        lags = _safe_nlags(n, prefer=prefer_lags)

        # ACF
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_acf(series, lags=lags, ax=ax)
        ax.set_title(f"{title_prefix} — ACF (lags={lags})")
        fig.tight_layout()
        fig.savefig(acf_path, dpi=150)
        plt.close(fig)

        # PACF
        fig, ax = plt.subplots(figsize=(8, 4))
        plot_pacf(series, lags=lags, method="ywm", ax=ax)
        ax.set_title(f"{title_prefix} — PACF (lags={lags})")
        fig.tight_layout()
        fig.savefig(pacf_path, dpi=150)
        plt.close(fig)

    # Hourly
    try:
        df = pd.read_csv(OUTPUT_FILES["Hourly"])
        s = pd.to_numeric(df["Total sales"], errors="coerce").dropna().values
        _save_acf_pacf_named(
            s, "Hourly Total sales",
            acf_path=PLOTS_DIR / "acf_total_sales_hourly.png",
            pacf_path=PLOTS_DIR / "pacf_total_sales_hourly.png"
        )
    except Exception as e:
        print(f"[AVISO] No se pudieron generar los plots EDA para Hourly: {e}")

    # Daily
    try:
        df = pd.read_csv(OUTPUT_FILES["Daily"])
        s = pd.to_numeric(df["Total sales"], errors="coerce").dropna().values
        _save_acf_pacf_named(
            s, "Daily Total sales",
            acf_path=PLOTS_DIR / "acf_total_sales_daily.png",
            pacf_path=PLOTS_DIR / "pacf_total_sales_daily.png"
        )
    except Exception as e:
        print(f"[AVISO] No se pudieron generar los plots EDA para Daily: {e}")

    # Weekly
    try:
        df = pd.read_csv(OUTPUT_FILES["Weekly"])
        s = pd.to_numeric(df["Total sales"], errors="coerce").dropna().values
        _save_acf_pacf_named(
            s, "Weekly Total sales",
            acf_path=PLOTS_DIR / "acf_total_sales_weekly.png",
            pacf_path=PLOTS_DIR / "pacf_total_sales_weekly.png"
        )
    except Exception as e:
        print(f"[AVISO] No se pudieron generar los plots EDA para Weekly: {e}")

    # Monthly
    try:
        df = pd.read_csv(OUTPUT_FILES["Monthly"])
        s = pd.to_numeric(df["Total sales"], errors="coerce").dropna().values
        _save_acf_pacf_named(
            s, "Monthly Total sales",
            acf_path=PLOTS_DIR / "acf_total_sales_monthly.png",
            pacf_path=PLOTS_DIR / "pacf_total_sales_monthly.png"
        )
    except Exception as e:
        print(f"[AVISO] No se pudieron generar los plots EDA para Monthly: {e}")

    return cleaned


# -----------------------------
# Main
# -----------------------------
if __name__ == "__main__":
    try:
        run_processing()
    except Exception as e:
        print(f"[ERROR] data_processing falló: {e}")
        sys.exit(1)