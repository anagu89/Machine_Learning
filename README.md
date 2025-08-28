# 💊 Pharma Sales Forecasting & Analytics  

Proyecto de Data Science aplicado al sector farmacéutico.  
Analizamos y predecimos ventas de medicamentos usando modelos de Machine Learning y series temporales, con visualizaciones interactivas en Streamlit.  

---

## 📂 Estructura del repositorio

```plaintext
├── data/
│   ├── raw/              # Datos originales (Kaggle)
│   ├── processed/        # Datos procesados
│   ├── train/            # Train set
│   └── test/             # Test set
├── docs/
│   ├── artifacts/        # KPIs, rankings, métricas
│   ├── plots/            # Gráficas generadas
│   ├── negocio.pptx      # Presentación para negocio
│   └── ds.pptx           # Presentación técnica
├── notebooks/
│   ├── 01_Fuentes.ipynb
│   ├── 02_LimpiezaEDA.ipynb
│   └── 03_Entrenamiento_Evaluacion.ipynb
├── src/
│   ├── utils.py
│   ├── data_processing.py
│   ├── training.py
│   ├── evaluation.py
│   └── plots.py
├── app_streamlit/
│   └── app.py            # Dashboard interactivo
│   └── requirements.txt  # Dependencias
├── models/
│   └── final_model.pkl   # Modelo entrenado
├── memoria.md            # Informe técnico
└── README.md
```

---

## ⚙️ Instalación y ejecución

```bash
# 1. Clonar repo
git clone https://github.com/usuario/pharma-sales-forecasting.git
cd pharma-sales-forecasting

# 2. Crear entorno virtual y activar
python -m venv venv
source venv/bin/activate  # en Windows: venv\Scripts\activate

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Procesar datos, entrenar y evaluar
python src/data_processing.py
python src/training.py
python src/evaluation.py

# 5. Lanzar dashboard
streamlit run app_streamlit/app.py
```

---

## 📊 Modelos implementados  

- ARIMA y SARIMAX  
- Random Forest  
- Gradient Boosting  
- XGBoost  
- **CatBoost (ganador 🏆)**  
- Clustering KMeans  
- *(Prophet reservado para futuro)*  

---


## ✨ Resultados clave  

- MAPE < 2% en predicción de ventas (**CatBoost**).  
- Identificación de categorías dominantes vía análisis de Pareto.  
- Segmentación temporal y clustering para detectar patrones de consumo.  
- Dashboard interactivo con filtros de granularidad, fechas y visualizaciones de negocio.  

---

## 📌 Dataset  

[Kaggle: Pharma Sales Data](https://www.kaggle.com/datasets/milanzdravkovic/pharma-sales-data)  

---


## 📜 Licencia  

No license. 