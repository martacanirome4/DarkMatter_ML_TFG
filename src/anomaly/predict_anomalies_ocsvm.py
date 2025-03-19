import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import matplotlib.pyplot as plt

# Aplicar el modelo entrenado de One-Class SVM a los datos de test en busca de anomalías
# detect_anomalies_ocsvm.py
# archivo de datos de test: data/raw/unids_3F_beta_err_names.txt
# modelo entrenado: outputs/models/ocsvm_astro_model.joblib
# scaler: outputs/models/scaler_astro.joblib
# archivo de salida: outputs/predictions/anomalies_ocsvm_astro.txt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cargar datos de test
file_path = os.path.join(project_root, 'data', 'raw', 'unids_3F_beta_err_names.txt')
df_test = pd.read_csv(file_path, sep='\s+')
# Modificar cabecera de sigma_det a sigma
df_test.rename(columns={'sigma_det': 'sigma'}, inplace=True)
# Seleccionar columnas de interés
X_test = df_test[['E_peak', 'beta', 'sigma', 'beta_Rel']].copy()

# Cargar modelo y scaler
model_dir = os.path.join(project_root, 'outputs', 'models')
ocsvm = joblib.load(os.path.join(model_dir, 'ocsvm_astro_model.joblib'))
scaler = joblib.load(os.path.join(model_dir, 'scaler_astro.joblib'))

# Normalizar datos de test
X_test_scaled = scaler.transform(X_test)

# Predecir anomalías
y_pred = ocsvm.predict(X_test_scaled)
df_test['anomaly'] = y_pred
df_test.to_csv(os.path.join(project_root, 'outputs', 'anomalies', 'anomalies_ocsvm_astro.txt'), sep='\t', index=False)
print("Anomalías detectadas y guardadas en outputs/anomalies/anomalies_ocsvm_astro.txt")

# Contar anomalías detectadas
print(df_test['anomaly'].value_counts())

# Calcular scores de anomalía: score más negativo = más anómalo
anomaly_scores = ocsvm.decision_function(X_test_scaled)
df_test['anomaly_score'] = anomaly_scores

# Visualizar la distribución de scores de anomalía
plt.figure(figsize=(8, 5))
plt.hist(df_test['anomaly_score'], bins=30, edgecolor='black')
plt.xlabel('Score de Anomalía (OCSVM)')
plt.ylabel('Cantidad de UNIDs')
plt.title('Distribución de Scores de Anomalía - UNIDs')
plt.tight_layout()
plt.savefig(os.path.join(project_root, 'outputs', 'anomalies', 'anomaly_score_distribution.png'))
plt.close()

df_test.to_csv(
    os.path.join(project_root, 'outputs', 'anomalies', 'anomalies_ocsvm_astro.csv'),
    index=False
)
print("Resultados guardados con score en anomalies_ocsvm_astro.csv")

