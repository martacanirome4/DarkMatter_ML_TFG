# Detección de Materia Oscura en Datos del Telescopio Fermi-LAT

**Trabajo de Fin de Grado – Universidad CEU San Pablo (2025)**  
Aplicación de técnicas de *Machine Learning* para la identificación de posibles señales de materia oscura en fuentes no identificadas del catálogo Fermi-LAT.

[![Last commit](https://img.shields.io/github/last-commit/martacanirome4/DarkMatter_ML_TFG)](https://github.com/martacanirome4/DarkMatter_ML_TFG/commits/main)
[![Python](https://img.shields.io/badge/Python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![License](https://img.shields.io/badge/License-Academic-green.svg)](LICENSE)

---

## Contexto científico

El telescopio espacial **Fermi-LAT** detecta rayos gamma de alta energía emitidos por fenómenos cósmicos extremos. Desde su lanzamiento por la NASA en 2008, ha permitido catalogar miles de fuentes de radiación gamma. Sin embargo, un gran porcentaje de ellas —denominadas **UNIDs (fuentes no identificadas)— aún no tienen una clasificación clara.

La hipótesis de que la materia oscura esté compuesta por **partículas masivas débilmente interactuantes (WIMPs)** sugiere que algunas de estas fuentes podrían ser el resultado de aniquilaciones de materia oscura, generando rayos gamma detectables.

Este proyecto explora la capacidad del aprendizaje automático para detectar estas anomalías mediante un enfoque interdisciplinar que combina ciencia de datos, física de partículas y astrofísica computacional.

![cielo_gamma](https://github.com/user-attachments/assets/23e7d654-3dcf-4ecd-8306-c3f7b082ca30)
*Mapa del cielo en rayos gamma observado por Fermi-LAT*

---

## Objetivo

El objetivo principal es aplicar técnicas de **Machine Learning supervisado y no supervisado** para identificar candidatos a señales de materia oscura entre las fuentes no identificadas del catálogo Fermi-LAT (4FGL-DR3).

### Enfoque metodológico:

- **Entrenar modelos** con fuentes astrofísicas conocidas (púlsares, blazares, etc.)
- **Detectar anomalías** entre las fuentes no identificadas (UNIDs)
- **Comparar predicciones** con modelos existentes de referencia (ANN)
- **Priorizar fuentes prometedoras** para seguimiento observacional
- **Validar resultados** mediante análisis estadístico y visualización

---

## Metodología y herramientas

| Enfoque / Modelo         | Propósito                                  | Herramienta principal     |
|--------------------------|--------------------------------------------|----------------------------|
| **One-Class SVM (OCSVM)** | Detección de anomalías sin etiquetas       | scikit-learn, Pandas       |
| **Red neuronal (ANN)**    | Modelo supervisado de referencia           | Scikit-learn               |
| **Fusión OCSVM + ANN**    | Comparación y sinergia entre enfoques      | Análisis cruzado           |

### Características técnicas:
- Visualizaciones 2D y 3D
- Análisis estadísticos avanzados
- Selección de hiperparámetros mediante **Grid Search**
- Validación cruzada y métricas de rendimiento
- Análisis de la frontera de decisión aprendida por los modelos

---

## 📁 Estructura del repositorio

```
DarkMatter_ML_TFG/
│
├── TFG_codigo_final/        # Código principal y scripts
│   ├── data/                # Datos intermedios y transformados
│   │   ├── results/         # Resultados OCSVM y ANN organizados por tipo
│   │   ├── *.txt, *.h5      # Archivos de entrada y salida
│   └── notebooks/           # Jupyter Notebooks experimentales
│       ├── OCSVM_*.ipynb    # Modelos de detección de anomalías
│       └── ANN_*.ipynb      # Comparación con redes neuronales
│
├── notebooks/               #Jupter notebooks exploratorios, de procesamiento y experimentales iniciales
├── docs/
├── DR4/
├── outputs/
├── src/
├── requirements.txt         # Entorno reproducible (librerías)
└── README.md
```

---

## Instalación y ejecución

### Requisitos previos
- Python 3.9 o superior
- Git

### 1. Clonación del repositorio
```bash
git clone https://github.com/martacanirome4/DarkMatter_ML_TFG.git
cd DarkMatter_ML_TFG
```

### 2. Configuración del entorno
```bash
# Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # En Windows: venv\Scripts\activate

# Instalar dependencias
pip install -r requirements.txt
```

### 3. Estructura de ejecución recomendada

#### **Fase exploratoria** (notebooks/)
```bash
jupyter notebook notebooks/
```
- Ejecutar notebooks exploratorios para familiarizarse con los datos del catálogo Fermi-LAT
- Análisis preliminar de características y distribuciones

#### **Análisis principal** (codigo_final/notebooks/)
```bash
cd TFG_codigo_final
jupyter notebook notebooks/
```

**Orden de ejecución recomendado:**
1. **Notebooks OCSVM**: Modelos de detección de anomalías
   - Preparación de datos
   - Entrenamiento One-Class SVM
   - Evaluación y visualización de resultados
   - **Análisis comparativo**: Fusión de resultados OCSVM + ANN
   
2. **Notebooks ANN**: Redes neuronales de referencia
   - Implementación del modelo de Gammaldi et al. (2023)
   - Comparación con resultados publicados

---

## Resultados destacados

### Hiperfrontera de decisión One-Class SVM 2F
![ocsvm_optimal_boundary_2F](https://github.com/user-attachments/assets/34bb135e-0c99-48f6-b754-70c4c6ad5e01)

### Hiperfrontera de decisión One-Class SVM 4F
![ocsvm_4f_optimal_boundary](https://github.com/user-attachments/assets/af39faa8-bea0-40e2-b0c2-426c5018edc0)

### Comparación de modelos
![venn_2f_4f](https://github.com/user-attachments/assets/27a1544b-e5c5-40b3-b67a-df846f5d4277)
![venn_2f_vs_ann2f](https://github.com/user-attachments/assets/238e7e13-226c-49b1-82d6-8ef531f93159)

---

## 📖 Recursos científicos y referencias

### **Estudio de referencia**
- **Gammaldi et al., 2023** – *A search for dark matter among Fermi-LAT unidentified sources*  
  [📄 Artículo MNRAS](https://academic.oup.com/mnras/article/520/1/1348/6987092)

### 🛰️ **Datos y catálogos**
- **Catálogo Fermi-LAT NASA**:  
  [🔗 Fermi Science Support Center](https://fermi.gsfc.nasa.gov/ssc/data/access/)
- **4FGL**: Fourth Fermi-LAT Source Catalog
- **DR4**: Data Release 4

### 💻 **Código base**
- **Implementación ANN original**:  
  [🔗 ViviGamma/Fermi_LAT_unids_NN](https://github.com/ViviGamma/Fermi_LAT_unids_NN)

---

## Dependencias principales

```txt
scikit-learn>=1.0.0     # Algoritmos de Machine Learning
pandas>=1.3.0           # Manipulación de datos
numpy>=1.20.0           # Computación numérica
matplotlib>=3.5.0       # Visualizaciones básicas
seaborn>=0.11.0         # Visualizaciones estadísticas
jupyter>=1.0.0          # Entorno de notebooks
h5py>=3.0.0             # Manejo de archivos HDF5
astropy>=5.0.0          # Astronomía y coordenadas celestes
```

---

## Características del dataset

- **Fuentes totales**: ~5000 fuentes del catálogo 4FGL
- **UNIDs analizadas**: ~1000 fuentes no identificadas
- **Características por fuente**: 
  - Log(E_peak): Energía del pico espectral
  - Log(beta): Curvatura espectral
  - Log(sigma): Significancia estadística de detección
  - Log(beta_rel): Error relativo de la curvatura espectral

---

## Autoría

**Marta Canino Romero**  
Grado en Ingeniería en Sistemas de Información  
Universidad CEU San Pablo (Madrid, España) – TFG 2025  

📧 Contacto: martacaninoromero@gmail.com
🔗 GitHub: @martacanirome4

---

## Licencia y uso académico

Este repositorio se publica con **fines académicos**. Todo el contenido es original salvo donde se indique lo contrario.
---

## Enlaces útiles

- [📖 Documentación Fermi-LAT](https://fermi.gsfc.nasa.gov/ssc/data/analysis/documentation/)
- [🧮 Scikit-learn Documentation](https://scikit-learn.org/stable/)
- [🌌 Astrofísica de rayos gamma](https://heasarc.gsfc.nasa.gov/docs/objects.html)
- [🔬 Dark Matter searches](https://darkmachines.org/)

---

*Última actualización: Junio 2025*
