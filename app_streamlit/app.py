# app_streamlit/app.py
from __future__ import annotations

import json
from pathlib import Path
from datetime import date

import numpy as np
import pandas as pd
import streamlit as st
import altair as alt
import plotly.graph_objects as go

# =============================
# Rutas del proyecto
# =============================
BASE_DIR = Path(__file__).resolve().parents[1]
DATA_DIR = BASE_DIR / "data"
PROC_DIR = DATA_DIR / "processed"
TEST_DIR = DATA_DIR / "test"
MODELS_DIR = BASE_DIR / "models"
DOCS_DIR = BASE_DIR / "docs"
ARTI_DIR = DOCS_DIR / "artifacts"
PLOTS_DIR = DOCS_DIR / "plots"

# =============================
# Utils de carga
# =============================
@st.cache_data(show_spinner=False)
def _safe_read_csv(path: Path) -> pd.DataFrame | None:
    try:
        if path.exists():
            return pd.read_csv(path)
    except Exception as e:
        st.warning(f"No se pudo leer {path.name}: {e}")
    return None

@st.cache_data(show_spinner=False)
def load_processed(level: str) -> pd.DataFrame | None:
    name = f"sales{level.lower()}_clean.csv"
    return _safe_read_csv(PROC_DIR / name)

@st.cache_data(show_spinner=False)
def load_metrics_summary() -> pd.DataFrame | None:
    path = ARTI_DIR / "metrics_summary.csv"
    if path.exists():
        # Mantener índice del CSV (nombre del modelo)
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data(show_spinner=False)
def load_eval_metrics() -> pd.DataFrame | None:
    path = ARTI_DIR / "eval_metrics.csv"
    if path.exists():
        # Mantener índice "final_model" para poder renombrarlo
        return pd.read_csv(path, index_col=0)
    return None

@st.cache_data(show_spinner=False)
def load_y_true_pred() -> pd.DataFrame | None:
    return _safe_read_csv(ARTI_DIR / "y_true_pred.csv")

@st.cache_data(show_spinner=False)
def load_kpis() -> pd.DataFrame | None:
    return _safe_read_csv(ARTI_DIR / "kpis_por_dataset.csv")

@st.cache_data(show_spinner=False)
def load_features_json() -> dict | None:
    path = ARTI_DIR / "features.json"
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception as e:
            st.warning(f"No se pudo leer features.json: {e}")
    return None

def get_date_bounds(df: pd.DataFrame) -> tuple[date, date] | None:
    if "datum" in df.columns:
        d = pd.to_datetime(df["datum"], errors="coerce").dropna()
        if len(d) > 0:
            return d.min().date(), d.max().date()
    if "fecha" in df.columns:
        d = pd.to_datetime(df["fecha"], errors="coerce").dropna()
        if len(d) > 0:
            return d.min().date(), d.max().date()
    return None

def robust_date_input(label: str, min_d: date, max_d: date, key: str = "rango_fechas") -> tuple[date, date]:
    """
    Devuelve (start_date, end_date) aunque Streamlit retorne una fecha única.
    """
    default_range = (min_d, max_d)
    sel = st.sidebar.date_input(label, value=default_range, min_value=min_d, max_value=max_d, key=key)
    if isinstance(sel, (list, tuple)) and len(sel) == 2:
        start, end = sel
    else:
        start = sel if isinstance(sel, date) else min_d
        end = sel if isinstance(sel, date) else max_d
    if hasattr(start, "date"): start = start.date()
    if hasattr(end, "date"):   end = end.date()
    if start > end:
        start, end = end, start
    return start, end

def filter_df_by_date(df: pd.DataFrame, start: date, end: date) -> pd.DataFrame:
    if "datum" in df.columns:
        d = df.copy()
        d["__dt"] = pd.to_datetime(d["datum"], errors="coerce")
        m = (d["__dt"].dt.date >= start) & (d["__dt"].dt.date <= end)
        return d.loc[m].drop(columns="__dt")
    if "fecha" in df.columns:
        d = df.copy()
        d["__dt"] = pd.to_datetime(d["fecha"], errors="coerce")
        m = (d["__dt"].dt.date >= start) & (d["__dt"].dt.date <= end)
        return d.loc[m].drop(columns="__dt")
    return df

def time_series_chart(df: pd.DataFrame, col_date: str, col_value: str, color: str, title: str):
    d = df.copy()
    d[col_date] = pd.to_datetime(d[col_date], errors="coerce")
    d = d.dropna(subset=[col_date])
    line = (
        alt.Chart(d)
        .mark_line()
        .encode(
            x=alt.X(col_date, title="Fecha"),
            y=alt.Y(col_value, title=col_value),
            color=alt.value(color),
            tooltip=[col_date, alt.Tooltip(col_value, format=",.2f")]
        )
        .properties(title=title, height=300)
    )
    st.altair_chart(line, use_container_width=True)

def bar_chart(df: pd.DataFrame, x_col: str, y_col: str, title: str):
    c = (
        alt.Chart(df)
        .mark_bar()
        .encode(
            x=alt.X(x_col, sort="-y", title=x_col),
            y=alt.Y(y_col, title=y_col),
            tooltip=[x_col, alt.Tooltip(y_col, format=",.2f")]
        )
        .properties(title=title, height=max(220, 24 * max(1, len(df))))
    )
    st.altair_chart(c, use_container_width=True)

def try_show_image(path: Path, caption: str | None = None):
    if path.exists():
        # Compatibilidad con versiones previas de Streamlit
        st.image(str(path), caption=caption or path.name, use_column_width=True)

# =============================
# Layout
# =============================
st.set_page_config(page_title="💊 Pharma Sales Forecasting", layout="wide")
st.title("💊 Pharma Sales Forecasting")

# Sidebar controles
st.sidebar.title("⚙️ Controles")
levels = ["Daily", "Weekly", "Monthly", "Hourly"]
level = st.sidebar.selectbox("Granularidad", levels, index=0)

# Cargar dataset seleccionado
df_level = load_processed(level)
if df_level is None or df_level.empty:
    st.error(f"No se pudo cargar el dataset procesado para {level}. Ejecuta primero `python src/data_processing.py`.")
    st.stop()

# Rango de fechas
bounds = get_date_bounds(df_level)
if bounds:
    min_date, max_date = bounds
    start_date, end_date = robust_date_input("Rango de fechas", min_date, max_date, key=f"rango_{level}")
    df_level_f = filter_df_by_date(df_level, start_date, end_date)
else:
    st.sidebar.info("No se detectaron fechas en el dataset. Se mostrará todo el contenido.")
    df_level_f = df_level.copy()

# =============================
# Tabs
# =============================
tabs = st.tabs([
    "🏠 Visión general",
    "📏 Evaluación y predicciones",
    "🤖 Modelo",
    "🧠 Importancias / Explicabilidad",
    "🧩 Clustering (Daily)",
    "📈 Patrones",
    "🧪 Categorías",
    "📊 Galería de plots",
    "🧾 Datos",
])

# =============================
# 🏠 Visión general
# =============================
with tabs[0]:
    st.subheader(f"🏠 Visión general — {level}")

    # KPIs
    kpis_df = load_kpis()
    if kpis_df is not None and not kpis_df.empty:
        c1, c2, c3, c4 = st.columns(4)
        try:
            row = kpis_df.loc[kpis_df["dataset"] == level].iloc[0]
            c1.metric("Filas", f"{int(row['n_filas']):,}".replace(",", "."))
            c2.metric("Fecha min", str(pd.to_datetime(row["fecha_min"]).date()))
            c3.metric("Fecha max", str(pd.to_datetime(row["fecha_max"]).date()))
            c4.metric("Media Total sales", f"{row['media_total_sales']:.2f}")
        except Exception:
            st.info("KPIs no disponibles para este nivel.")
    else:
        st.info("No hay KPIs guardados (docs/artifacts/kpis_por_dataset.csv).")

    # Serie temporal
    date_col = "datum" if "datum" in df_level_f.columns else ("fecha" if "fecha" in df_level_f.columns else None)
    if date_col and "Total sales" in df_level_f.columns:
        st.markdown("**Serie temporal (Total sales)**")
        time_series_chart(df_level_f, date_col, "Total sales", "#1f77b4", f"{level} — Total sales")
    else:
        st.warning("No se encontraron columnas para graficar la serie (se requieren 'datum/fecha' y 'Total sales').")

# =============================
# 📏 Evaluación y predicciones
# =============================
with tabs[1]:
    st.subheader("📏 Evaluación y predicciones (test)")
    eval_df = load_eval_metrics()
    yp = load_y_true_pred()
    ms = load_metrics_summary()

    # Renombrar la fila "final_model" por el nombre del mejor modelo (si lo tenemos)
    display_eval = None
    best_label = None
    if ms is not None and not ms.empty:
        try:
            best_label = ms["MAPE"].astype(float).idxmin()
        except Exception:
            best_label = None

    if eval_df is not None and not eval_df.empty:
        display_eval = eval_df.copy()
        # si el índice tiene una sola fila (habitual), renómbrala
        try:
            if best_label is not None and len(display_eval.index) == 1:
                display_eval.index = [best_label]
        except Exception:
            pass

        st.markdown("**Métricas (test)**")
        st.dataframe(
            display_eval.style.format({"MAPE": "{:.3f}", "MAE": "{:.3f}", "MSE": "{:.3f}", "R2": "{:.3f}"}),
            use_container_width=True
        )
    else:
        st.info("No se encontró docs/artifacts/eval_metrics.csv. Ejecuta `python src/evaluation.py`.")

    if yp is not None and not yp.empty:
        # Filtrado por rango si procede
        bounds_yp = get_date_bounds(yp.rename(columns={"fecha": "datum"}))
        if bounds_yp and 'start_date' in locals():
            dfp = filter_df_by_date(yp.rename(columns={"fecha": "datum"}), start_date, end_date).rename(columns={"datum": "fecha"})
        else:
            dfp = yp.copy()

        # Gráfica Plotly: y_true = naranja, x con título "Fecha"
        if "y_true" in dfp.columns:
            st.markdown("**Real vs Predicción**")
            dfp_plot = dfp.copy()
            dfp_plot["fecha"] = pd.to_datetime(dfp_plot["fecha"], errors="coerce")

            fig = go.Figure()
            # y_true en naranja
            fig.add_trace(go.Scatter(
                x=dfp_plot["fecha"], y=dfp_plot["y_true"],
                mode="lines", name="Real", line=dict(color="#ff7f0e")
            ))
            # y_pred (color por defecto)
            fig.add_trace(go.Scatter(
                x=dfp_plot["fecha"], y=dfp_plot["y_pred"],
                mode="lines", name="Predicción"
            ))

            fig.update_layout(
                height=360,
                margin=dict(l=40, r=20, t=20, b=40),
                xaxis_title="Fecha",
                yaxis_title="Ventas",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.markdown("**Predicción (sin y_true en test)**")
            time_series_chart(dfp, "fecha", "y_pred", "#ff7f0e", "Predicción")
    else:
        st.info("No se encontró docs/artifacts/y_true_pred.csv. Ejecuta `python src/evaluation.py`.")

# =============================
# 🤖 Modelo (sustituye a Comparativa)
# =============================
with tabs[2]:
    st.subheader("🤖 Modelo")

    ms = load_metrics_summary()
    if ms is not None and not ms.empty:
        # Determinar mejor por MAPE
        try:
            ms_disp = ms.copy()
            st.markdown("**Resumen de modelos**")
            st.dataframe(ms_disp, use_container_width=True)

            best_idx = ms_disp["MAPE"].astype(float).idxmin()
            best_row = ms_disp.loc[best_idx]
            st.success(
                f"**Mejor modelo:** {best_idx}  |  "
                f"MAPE={best_row['MAPE']:.3f}  MAE={best_row['MAE']:.3f}  "
                f"MSE={best_row['MSE']:.3f}  R2={best_row['R2']:.3f}"
            )
        except Exception:
            st.info("No se pudo determinar el mejor modelo por MAPE.")
    else:
        st.info("No hay docs/artifacts/metrics_summary.csv. Ejecuta `python src/training.py` para generarlo.")

    # features.json si existe
    feats_info = load_features_json()
    if feats_info:
        st.markdown("**Features usadas por el modelo final (features.json)**")
        st.code(json.dumps(feats_info, indent=2, ensure_ascii=False), language="json")
    else:
        st.caption("Sugerencia: exporta docs/artifacts/features.json con el orden de feat_cols del modelo final.")

    # Mostrar modelo final guardado si existe (con nombre)
    final_model_path = MODELS_DIR / "final_model.pkl"
    if final_model_path.exists():
        ms2 = load_metrics_summary()
        if ms2 is not None and not ms2.empty:
            try:
                best_idx_2 = ms2["MAPE"].astype(float).idxmin()
                st.info(f"🏆 **Modelo final elegido: {best_idx_2}** guardado en `{final_model_path.name}`")
            except Exception:
                st.info(f"🏆 **Modelo final elegido** guardado en `{final_model_path.name}`")
        else:
            st.info(f"🏆 **Modelo final elegido** guardado en `{final_model_path.name}`")
    else:
        st.warning("No se encontró `final_model.pkl`. Ejecuta `python src/training.py` para generarlo.")

# =============================
# 🧠 Importancias / Explicabilidad
# =============================
with tabs[3]:
    st.subheader("🧠 Importancias / Explicabilidad")

    feats_info = load_features_json()
    if feats_info:
        st.code(json.dumps(feats_info, indent=2, ensure_ascii=False), language="json")

    # PNGs de importancias si existen
    imp_imgs = [
        PLOTS_DIR / "feature_importances_RF_best.png",
        PLOTS_DIR / "feature_importances_GBR_best.png",
        PLOTS_DIR / "feature_importances_XGB_best.png",
        PLOTS_DIR / "feature_importances_CAT_best.png",
    ]
    any_img = False
    for p in imp_imgs:
        if p.exists():
            any_img = True
            try_show_image(p, caption=p.name)
    if not any_img:
        st.info("No se encontraron imágenes de importancias en docs/plots/. Se generan al entrenar (training.py).")

# =============================
# 🧩 Clustering (Daily)
# =============================
with tabs[4]:
    st.subheader("🧩 Clustering (sólo Daily)")
    if level != "Daily":
        st.info("Cambia a 'Daily' en la barra lateral para ver resultados de clustering.")
    else:
        cluster_csv = ARTI_DIR / "daily_clusters.csv"
        cluster_metric_csv = ARTI_DIR / "cluster_metric.csv"
        if cluster_csv.exists():
            dcl = pd.read_csv(cluster_csv)
            if "fecha" in dcl.columns and 'start_date' in locals():
                dcl = filter_df_by_date(dcl.rename(columns={"fecha": "datum"}), start_date, end_date).rename(columns={"datum": "fecha"})
            st.markdown("**Asignaciones de cluster (muestra)**")
            st.dataframe(dcl.head(100), use_container_width=True)
        else:
            st.info("No se encontró docs/artifacts/daily_clusters.csv. Ejecuta `python src/training.py`.")

        if cluster_metric_csv.exists():
            st.markdown("**Métrica de clustering (Silhouette & k)**")
            st.dataframe(pd.read_csv(cluster_metric_csv, index_col=0), use_container_width=True)

        try_show_image(PLOTS_DIR / "kmeans_pca_scatter.png", caption="PCA scatter por cluster")
        try_show_image(PLOTS_DIR / "kmeans_features_heatmap.png", caption="Heatmap medias por cluster")
        try_show_image(PLOTS_DIR / "kmeans_cluster_sizes.png", caption="Tamaños de cluster")

# =============================
# 📈 Patrones (estacionalidad y semanal — Daily)
# =============================
with tabs[5]:
    st.subheader("📈 Patrones (Daily)")

    if level != "Daily":
        st.info("Cambia a 'Daily' para ver estacionalidad y patrón semanal.")
    else:
        df_daily = df_level_f.copy()
        if "datum" in df_daily.columns and "Total sales" in df_daily.columns:
            df_daily["datum"] = pd.to_datetime(df_daily["datum"], errors="coerce")
            df_daily = df_daily.dropna(subset=["datum"])
            df_daily["Month"] = df_daily["datum"].dt.month
            df_daily["Weekday"] = df_daily["datum"].dt.dayofweek + 1

            # Boxplot por mes (Altair workaround usando rule + tick)
            st.markdown("**Estacionalidad intra–anual (boxplot por mes)**")
            stats_m = (
                df_daily.groupby("Month")["Total sales"]
                .agg(q1=lambda x: np.percentile(x, 25),
                     q3=lambda x: np.percentile(x, 75),
                     med="median",
                     min="min",
                     max="max").reset_index()
            )
            base = alt.Chart(stats_m).encode(x=alt.X("Month:O", title="Mes"))
            rule = base.mark_rule().encode(y="q1:Q", y2="q3:Q", tooltip=["Month","q1","q3","med","min","max"])
            tick_med = base.mark_tick(thickness=2, size=20).encode(y="med:Q")
            whisker_low = base.mark_rule().encode(y="min:Q", y2="q1:Q")
            whisker_high = base.mark_rule().encode(y="q3:Q", y2="max:Q")
            st.altair_chart(rule + whisker_low + whisker_high + tick_med, use_container_width=True)

            # Patrón semanal (medias por día de la semana)
            st.markdown("**Patrón semanal (media por día de la semana)**")
            avg_w = df_daily.groupby("Weekday")["Total sales"].mean().reset_index()
            avg_w["Weekday"] = avg_w["Weekday"].astype(int)
            bar_chart(avg_w, "Weekday", "Total sales", "Media por día de la semana")
        else:
            st.warning("No se encuentran columnas necesarias ('datum', 'Total sales') en Daily.")

# =============================
# 🧪 Categorías (Daily)
# =============================
with tabs[6]:
    st.subheader("🧪 Categorías (Daily)")
    if level != "Daily":
        st.info("Cambia a 'Daily' para ver Pareto, Stacked y ranking de categorías.")
    else:
        # Intento de usar artefactos existentes (ranking_categorias_daily.csv)
        rank_path = ARTI_DIR / "ranking_categorias_daily.csv"
        if rank_path.exists():
            ranking = pd.read_csv(rank_path).reset_index().rename(columns={"index": "Categoria", "ventas": "Ventas"})
        else:
            # Recalcular rápido a partir del processed daily
            df_daily = load_processed("Daily")
            if df_daily is not None:
                atc_cols = [c for c in ["M01AB","M01AE","N02BA","N02BE","N05B","N05C","R03","R06"] if c in df_daily.columns]
                if atc_cols:
                    tot = df_daily[atc_cols].sum().sort_values(ascending=False)
                    ranking = tot.to_frame("Ventas").reset_index().rename(columns={"index": "Categoria"})
                else:
                    ranking = None
            else:
                ranking = None

        if ranking is None or ranking.empty:
            st.info("No hay columnas ATC para construir los gráficos de categorías.")
        else:
            # Ranking tabla
            st.markdown("**Ranking de categorías (tabla)**")
            st.dataframe(ranking, use_container_width=True)

            # Pareto
            st.markdown("**Pareto de categorías**")
            r = ranking.copy()
            r["Pct"] = r["Ventas"] / r["Ventas"].sum() * 100.0
            r["Acum"] = r["Pct"].cumsum()
            bars = (
                alt.Chart(r)
                .mark_bar()
                .encode(
                    x=alt.X("Categoria:N", sort="-y"),
                    y=alt.Y("Ventas:Q"),
                    tooltip=["Categoria","Ventas", alt.Tooltip("Pct:Q", format=".2f"), alt.Tooltip("Acum:Q", format=".2f")]
                )
            )
            line = (
                alt.Chart(r)
                .mark_line(point=True)
                .encode(
                    x=alt.X("Categoria:N", sort="-y"),
                    y=alt.Y("Acum:Q", axis=alt.Axis(title="% acumulado", grid=False)),
                    color=alt.value("#d62728")
                )
            )
            st.altair_chart(alt.layer(bars, line).resolve_scale(y='independent').properties(height=320), use_container_width=True)

            # Stacked Top N en el tiempo
            st.markdown("**Stacked Top N categorías en el tiempo**")
            top_n = int(st.number_input("Top N", min_value=2, max_value=8, value=5, step=1))
            df_daily = load_processed("Daily")
            df_daily["datum"] = pd.to_datetime(df_daily["datum"], errors="coerce")
            df_daily = df_daily.dropna(subset=["datum"])
            top_cols = list(ranking.sort_values("Ventas", ascending=False).head(top_n)["Categoria"])
            present = [c for c in top_cols if c in df_daily.columns]
            if present:
                dlong = df_daily[["datum"] + present].melt(id_vars="datum", var_name="Categoria", value_name="Ventas")
                dlong = dlong.dropna()
                area = (
                    alt.Chart(dlong)
                    .mark_area()
                    .encode(
                        x="datum:T",
                        y=alt.Y("sum(Ventas):Q", stack="zero"),
                        color="Categoria:N",
                        tooltip=["datum:T", "Categoria:N", alt.Tooltip("Ventas:Q", format=",.2f")]
                    )
                    .properties(height=320)
                )
                st.altair_chart(area, use_container_width=True)
            else:
                st.info("No se encontraron columnas de las categorías Top N en el Daily procesado.")

# =============================
# 📊 Galería de plots (PNG existentes)
# =============================
with tabs[7]:
    st.subheader("📊 Galería de plots")

    # ACF/PACF por nivel
    acf_pacf_map = {
        "Daily":   ("acf_total_sales_daily.png",   "pacf_total_sales_daily.png"),
        "Weekly":  ("acf_total_sales_weekly.png",  "pacf_total_sales_weekly.png"),
        "Monthly": ("acf_total_sales_monthly.png", "pacf_total_sales_monthly.png"),
        "Hourly":  ("acf_total_sales_hourly.png",  "pacf_total_sales_hourly.png"),
    }
    if level in acf_pacf_map:
        acf_name, pacf_name = acf_pacf_map[level]
        try_show_image(PLOTS_DIR / acf_name, caption=acf_name)
        try_show_image(PLOTS_DIR / pacf_name, caption=pacf_name)

    # Otras figuras estándar si existen
    other_imgs = [
        PLOTS_DIR / "forecast_best_model_matplotlib.png",
        PLOTS_DIR / "model_comparison_mape.png",
        PLOTS_DIR / "ventas_por_categoria_barras.png",
        PLOTS_DIR / "ventas_por_categoria_pareto.png",
        PLOTS_DIR / "ventas_por_categoria_stacked.png",
    ]
    for p in other_imgs:
        try_show_image(p, caption=p.name)

# =============================
# 🧾 Datos (preview y descarga)
# =============================
with tabs[8]:
    st.subheader(f"🧾 Datos — {level}")
    st.dataframe(df_level_f.head(500), use_container_width=True)
    st.caption("Se muestran hasta 500 filas para visualización rápida.")

    st.download_button(
        "Descargar CSV filtrado",
        data=df_level_f.to_csv(index=False).encode("utf-8"),
        file_name=f"{level.lower()}_filtered.csv",
        mime="text/csv"
    )

# =============================
# Footer
# =============================
st.caption(f"Base dir: {BASE_DIR}")