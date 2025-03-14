# Búsqueda de Materia Oscura mediante Machine Learning en Fuentes Gamma No Identificadas del Fermi-LAT

Este repositorio contiene el código, los datos y la documentación asociados a mi proyecto de tesis. El proyecto se centra en la aplicación de técnicas de machine learning, en particular redes neuronales, para clasificar las fuentes gamma no identificadas del catálogo Fermi-LAT. El objetivo final es buscar posibles firmas de materia oscura (DM) dentro de estas fuentes, incorporando características sistemáticas como la significancia de detección y las incertidumbres en los parámetros espectrales.

## Referencias
Repositorio Original en GitHub:
ViviGamma/Fermi_LAT_unids_NN
Este repositorio proporciona el código original y la metodología en la que se basa este proyecto.

## Motivación
El catálogo Fermi-LAT contiene miles de fuentes gamma, muchas de las cuales aún no han sido identificadas. Si la materia oscura está compuesta por partículas masivas de interacción débil (WIMPs), un subconjunto de estas fuentes no identificadas podría originarse en procesos de aniquilación de DM. Esta tesis explora cómo el uso de machine learning, potenciado por características sistemáticas derivadas de datos observacionales, puede ayudar a identificar candidatas a materia oscura entre estas fuentes.

## Estructura del Repositorio
```bash
/mi-proyecto-ml-dark-matter/
│── data/                 
│   │── raw/              # Datos sin procesar
│   │── processed/        # Datos limpios y normalizados
│── notebooks/            # Jupyter Notebooks para análisis
│   │── 01_exploracion.ipynb  # Exploración de datos y visualización
│   │── 02_preprocesamiento.ipynb  # Limpieza, normalización y guardado
│   │── 03_entrenamiento_modelo.ipynb  # Entrenamiento de OneClassSVM
│── src/                  
│   │── data_processing/  
│   │   ├── clean_data.py  # Limpieza y eliminación de duplicados
│   │   ├── normalize_data.py  # Normalización de datos
│   │── models/           
│   │   ├── one_class_svm.py  # Definición del modelo OneClassSVM
│   │   ├── train_model.py  # Entrenamiento y evaluación
│── scripts/              
│   │── preprocess_data.py  # Automatiza preprocesamiento
│   │── train_model.py  # Automatiza entrenamiento
│── results/              
│   │── figures/          
│   │── reports/          
│── docs/                 
│   │── README.md         
│── requirements.txt      
│── setup.py              
│── .gitignore            
│── README.md             
```

# Licencia
Este proyecto se distribuye bajo la Licencia MIT.
