
# Detección de Materia Oscura en Datos del Telescopio Fermi-LAT

**Trabajo de Fin de Grado – Universidad CEU San Pablo (2025)**  
Aplicación de técnicas de *Machine Learning* para la identificación de posibles señales de materia oscura en fuentes no identificadas del catálogo Fermi-LAT.

[![CI](https://img.shields.io/github/actions/workflow/status/martacanirome4/DarkMatter_ML_TFG/ci.yml?branch=origin)](https://github.com/martacanirome4/DarkMatter_ML_TFG/actions)
[![Last commit](https://img.shields.io/github/last-commit/martacanirome4/DarkMatter_ML_TFG)](https://github.com/martacanirome4/dark-matter-api/commits/main)
---

## Contexto científico

El telescopio espacial Fermi-LAT detecta rayos gamma de alta energía emitidos por fenómenos cósmicos extremos. Desde su lanzamiento por la NASA en 2008, ha permitido catalogar miles de fuentes de radiación gamma. Sin embargo, un gran porcentaje de ellas —denominadas UNIDs (fuentes no identificadas)— aún no tienen una clasificación clara.

La hipótesis de que la materia oscura esté compuesta por partículas masivas débilmente interactuantes (WIMPs) sugiere que algunas de estas fuentes podrían ser el resultado de aniquilaciones de materia oscura, generando rayos gamma detectables.

Este proyecto explora la capacidad del aprendizaje automático para detectar estas anomalías mediante un enfoque interdisciplinar que combina ciencia de datos, física de partículas y astrofísica computacional.

![cielo_gamma](https://github.com/user-attachments/assets/23e7d654-3dcf-4ecd-8306-c3f7b082ca30)

---

## Objetivo

El objetivo de este TFG es aplicar técnicas de Machine Learning (ML) supervisado y no supervisado para identificar candidatos a señales de materia oscura entre las fuentes no identificadas del catálogo Fermi-LAT (4FGL y DR4).

### Enfoque general:

- Entrenar modelos con fuentes astrofísicas conocidas.
- Detectar anomalías entre las fuentes no identificadas (UNIDs).
- Comparar predicciones con modelos existentes de referencia (ANN).
- Priorizar las fuentes más prometedoras para seguimiento observacional.

---

## Metodología y herramientas

| Enfoque / Modelo         | Propósito                                  | Herramienta principal     |
|--------------------------|--------------------------------------------|----------------------------|
| One-Class SVM (OCSVM)    | Detección de anomalías sin etiquetas       | scikit-learn, Pandas       |
| Red neuronal (ANN)       | Modelo supervisado de referencia           | Scikit-learn               |
| Fusión OCSVM + ANN       | Comparación y sinergia entre enfoques      | Análisis cruzado           |

Incluye visualizaciones 3D, análisis estadísticos y selección de hiperparámetros mediante grid search.

---

## Estructura del repositorio

```
DarkMatter_ML_TFG/
│
├── codigo_final/            # Código principal y scripts
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

## Recursos científicos

- Estudio de referencia ANN:  
  Gammaldi et al., 2023 – A search for dark matter among Fermi-LAT unidentified sources  
  [Artículo MNRAS](https://academic.oup.com/mnras/article/520/1/1348/6987092)

- Catálogo Fermi-LAT NASA:  
  [Fermi Science Support Center](https://fermi.gsfc.nasa.gov/ssc/data/access/)

- Código ANN original:  
  [ViviGamma/Fermi_LAT_unids_NN](https://github.com/ViviGamma/Fermi_LAT_unids_NN)

---

## Autoría

Marta Canino Romero  
Grado en Ingeniería en Sistemas de Información  
Universidad CEU San Pablo (Madrid, España) – TFG 2025  
[GitHub: @martacanirome4](https://github.com/martacanirome4)

---

## Licencia

Este repositorio se publica con fines académicos. Todo el contenido es original salvo donde se indique lo contrario.
