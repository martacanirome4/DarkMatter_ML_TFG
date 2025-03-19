# 💫 Detección de Materia Oscura en Datos del Telescopio Fermi-LAT


Este proyecto tiene como objetivo identificar **posibles fuentes de materia oscura** dentro del conjunto de **fuentes no identificadas (UNIDs)** del catálogo **Fermi-LAT** de la NASA, mediante técnicas de **aprendizaje automático supervisado y detección de anomalías**.

---

## 🎯 Objetivo del Estudio
Desarrollar e implementar modelos que permitan distinguir entre fuentes astrofísicas conocidas y posibles señales de materia oscura, utilizando características espectrales extraídas de los datos del telescopio Fermi-LAT.  
El enfoque combina **clasificación supervisada** y **detección de novedades (novelty detection)** para lograr una identificación robusta de candidatas.

---

## Motivación
El catálogo Fermi-LAT contiene miles de fuentes gamma, muchas de las cuales aún no han sido identificadas. Si la materia oscura está compuesta por partículas masivas de interacción débil (WIMPs), un subconjunto de estas fuentes no identificadas podría originarse en procesos de aniquilación de DM. Esta tesis explora cómo el uso de machine learning, potenciado por características sistemáticas derivadas de datos observacionales, puede ayudar a identificar candidatas a materia oscura entre estas fuentes.

---

## 🧩 Metodología

1. **Clasificación Supervisada (Random Forest)**  
   Entrenado sobre fuentes astrofísicas identificadas y simulaciones de materia oscura → asigna **probabilidad de ser DM** a cada fuente UNID.

2. **Validación Cruzada con Red Neuronal (ANN)**  
   Comparación de resultados del modelo RF con un modelo ANN de un estudio previo → intersección de candidatas.

3. **Detección de Anomalías (One-Class SVM)**  
   Entrenado solo sobre fuentes astrofísicas → calcula **índice de anomalía** de cada fuente UNID.

4. **Fusión de Resultados**  
   Combinación de **probabilidad de materia oscura** y **anomalía** para determinar las candidatas más probables a ser fuentes de materia oscura.

---

## ⚙️ Requisitos

- Python 3.x
- Entorno virtual recomendado (`.venv`)
- Librerías clave: `scikit-learn`, `numpy`, `pandas`, `matplotlib`, `seaborn`, `joblib`

Instalar requisitos:
```bash
pip install -r requirements.txt
```

---

## Estructura del Proyecto
```bash
DarkMatter_ML_TFG/
│
├── data/
│   ├── raw/
│   │   ├── XY_bal_log_Rel.txt                  # Datos etiquetados (astro + DM simulada)
│   │   ├── unids_3F_beta_err_names.txt         # Datos de fuentes UNIDs
│   │   └── unids_DM_std_proba_...txt           # Resultados previos ANN
│   ├── processed/                              # Datos procesados para modelos
│   ├── 4GL_catalog/                            # Otros catálogos de referencia
│   └── results/                                # Resultados generados por modelos
│
├── docs/
│   └── references/                             # Documentación y fuentes bibliográficas
│
├── notebooks/
│   ├── 4F_ANN/                                 # Notebook ANN previo
│   ├── unids/                                  # Análisis de fuentes UNIDs
│   └── XY_bal_log_Rel/                         # EDA y análisis de datos etiquetados
│
├── outputs/
│   ├── predictions/                            # Predicciones sobre UNIDs
│   ├── compare/                                # Comparación RF vs ANN
│   ├── anomalies/                              # Resultados anomalías
│   └── models/                                 # Modelos entrenados (RF, OCSVM)
│
├── src/
│   ├── models/
│   │   └── classifier_rf.py                    # Entrena y aplica Random Forest
│   ├── predict/
│   │   └── predict_unids.py                    # Aplica RF a UNIDs
│   ├── compare/
│   │   └── compare_rf_ann_unids.py             # Comparación RF vs ANN
│   └── anomaly/
│       └── detect_anomalies_ocsvm.py           # Detección de anomalías (One-Class SVM)
│
├── requirements.txt
└── README.md
```

---

## 🚀 Cómo ejecutar scripts
1. **Activar entorno virtual:**  
    ```bash
    source .venv/bin/activate
    ```
2. **Entrenar Random Forest + aplicar a UNIDs:**  
    ```bash
    python src/models/classifier_rf.py
    ```
3. **Comparar resultados con ANN:**  
    ```bash
    python src/compare/compare_rf_ann_unids.py
    ```
4. **Detección de anomalías (One-Class SVM)::**  
    ```bash
    python src/anomaly/detect_anomalies_ocsvm.py
    ```

---

## 📚 Créditos y Recursos

- Catálogo Fermi-LAT NASA: [Enlace oficial](https://fermi.gsfc.nasa.gov/ssc/data/access/)
- Estudio original ANN:  
  *Gammaldi, V., Zaldívar, B., Sánchez-Conde, M. A., & Coronado-Blázquez, J. (2023). A search for dark matter among Fermi-LAT unidentified sources with systematic features in machine learning.*  
  [MNRAS, 520(1), 1348–1365](https://academic.oup.com/mnras/article/520/1/1348/6987092)
- Repositorio código ANN original:  
  [https://github.com/ViviGamma/Fermi_LAT_unids_NN](https://github.com/ViviGamma/Fermi_LAT_unids_NN)
- Desarrollado por: **Marta Canino Romero** – TFG Ingeniería Informática 2025 (Universidad CEU San Pablo, Madrid)

---

## Referencias
Repositorio Original en GitHub:
ViviGamma/Fermi_LAT_unids_NN
Este repositorio proporciona el código original y la metodología en la que se basa este proyecto.

---
