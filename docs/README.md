# 💫 Detección de Materia Oscura en Datos del Telescopio Fermi-LAT

**🚀 Identificación de posibles fuentes de materia oscura (DM) entre fuentes no identificadas (UNIDs) del catálogo Fermi-LAT de la NASA, usando machine learning supervisado y detección de anomalías.**

![galactic_haze_Fermi](https://github.com/user-attachments/assets/5a258831-133b-43bf-b4fd-4bc7c17b7206)

> Imagen: Bruma galáctica vista por Planck y 'burbujas' galácticas vistas por Fermi, obtenido por el telescopio Fermi-LAT – Fuente: NASA/DOE/Fermi LAT Collaboration (vía [Space.com](https://www.space.com/22466-nasa-fermi-telescope-photos-gamma-rays.html))

---

## 🌌 Motivación

El telescopio espacial Fermi-LAT detecta rayos gamma, y muchas de sus fuentes aún no están clasificadas. Si la materia oscura está formada por WIMPs (partículas masivas de interacción débil), es posible que parte de estas **UNIDs** tenga origen en **procesos de aniquilación de DM**.  
Este proyecto explora cómo **ML** puede ayudar a identificar candidatas, mezclando ciencia y tecnología.

---

## 🎯 Objetivo

Desarrollar modelos que distingan entre fuentes astrofísicas y posibles señales de DM, usando características espectrales derivadas del catálogo Fermi-LAT.  
El enfoque combina **clasificación supervisada** + **detección de anomalías**, y **fusión de resultados**.

---

## ⚙️ Metodología

| Técnica                  | Objetivo                                    | Herramienta         |
|-------------------------|---------------------------------------------|---------------------|
| Red Neuronal (ANN)      | Validar RF con resultados de otro estudio   | Código externo ANN  |
| One-Class SVM (OCSVM)   | Calcular anomalía de cada UNID              | scikit-learn        |
| Fusión ANN + OCSVM       | Determinar candidatas con alta probabilidad | Modelos combinados  |

---

## 🛰️ Curiosidades Astrofísicas y Tecnológicas

- **Materia Oscura** compone ~27% del universo, pero no emite luz. Solo se detecta por su influencia gravitatoria.
- Las WIMPs son candidatas a DM → podrían generar rayos gamma si se aniquilan, y **Fermi-LAT** busca justamente eso.
- **Fermi-LAT** es un satélite lanzado por NASA en 2008, especializado en detectar rayos gamma de alta energía.
- En ML, este campo se llama **astroinformática**: datos masivos + inteligencia artificial para estudiar el cosmos. 💫🧠
- El reto es que la **DM no tiene etiqueta**: aquí entra la detección de anomalías.

---

## 📚 Recursos Científicos

- 📄 Estudio ANN original:  
  *Gammaldi et al. (2023), MNRAS*  
  [Artículo completo](https://academic.oup.com/mnras/article/520/1/1348/6987092)

- 🔭 Catálogo Fermi-LAT NASA:  
  [Acceso oficial](https://fermi.gsfc.nasa.gov/ssc/data/access/)

- 💻 Código ANN original:  
  [ViviGamma/Fermi_LAT_unids_NN](https://github.com/ViviGamma/Fermi_LAT_unids_NN)

---

## ✍️ Autoría

Desarrollado por:  
**Marta Canino Romero** – TFG Ingeniería Informática 2025  
Universidad CEU San Pablo, Madrid 🇪🇸  
[GitHub](https://github.com/martacanirome4)
