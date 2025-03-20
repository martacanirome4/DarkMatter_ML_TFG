# Detectar anomalías en datos astro con One-Class SVM
# Entrenamos el modelo One-Class SVM con datos astrofísicos para detectar anomalías en nuevas observaciones.
# Guardamos el modelo y el escalador para normalizar los datos de entrada.
# src/anomaly/detect_anomalies_ocsvm.py
# detect_anomalies_ocsvm.py
import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(project_root, 'data', 'raw', 'XY_bal_log_Rel.txt')

df = pd.read_csv(file_path, sep='\s+')
df_astro = df[df['0,1=astro,DM'] == 0.0]
X_astro = df_astro[['E_peak', 'beta', 'sigma', 'beta_Rel']].copy()

print(f"Total fuentes ASTRO: {len(X_astro)}")
print(X_astro.head())

# === Normalización ===
scaler = StandardScaler()
X_astro_scaled = scaler.fit_transform(X_astro)

# === Entrenamiento One-Class SVM ===
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
ocsvm.fit(X_astro_scaled)
print("Modelo One-Class SVM entrenado con datos ASTRO")

timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
model_dir = os.path.join(project_root, 'outputs', 'models')
os.makedirs(model_dir, exist_ok=True)

model_path = os.path.join(model_dir, f'ocsvm_astro_model_{timestamp}.joblib')
scaler_path = os.path.join(model_dir, f'scaler_astro_{timestamp}.joblib')

# === Guardar modelo y scaler ===
joblib.dump(ocsvm, model_path)
joblib.dump(scaler, scaler_path)
print(f"Modelo guardado en: {model_path}")
print(f"Scaler guardado en: {scaler_path}")

# === Visualizar distribución de puntuaciones ===
scores = ocsvm.decision_function(X_astro_scaled)  # Valores más bajos = más anomalía
plt.figure(figsize=(10, 6))
plt.hist(scores, bins=30, edgecolor='black', color='steelblue')
plt.xlabel('Puntuación OCSVM (decision_function)', fontsize=12)
plt.ylabel('Número de fuentes ASTRO', fontsize=12)
plt.title(f'Distribución de Puntuaciones - One-Class SVM\n{timestamp}', fontsize=14, weight='bold')
plt.tight_layout()

# Guardar gráfico
plot_path = os.path.join(model_dir, f'ocsvm_scores_hist_{timestamp}.png')
plt.savefig(plot_path, dpi=300)
plt.close()
print(f"Histograma de puntuaciones guardado: {plot_path}")

# === Guardar log de entrenamiento ===
log_path = os.path.join(model_dir, f'ocsvm_training_log_{timestamp}.txt')
with open(log_path, 'w') as f:
    f.write(f"=== Entrenamiento One-Class SVM - Anomalías ASTRO ===\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write(f"Total fuentes ASTRO usadas: {len(X_astro)}\n")
    f.write("Parámetros del modelo:\n")
    f.write(f"- Kernel: {ocsvm.kernel}\n")
    f.write(f"- Gamma: {ocsvm.gamma}\n")
    f.write(f"- Nu (proporción de anomalías esperadas): {ocsvm.nu}\n\n")
    f.write("Archivos generados:\n")
    f.write(f"- Modelo OCSVM: {model_path}\n")
    f.write(f"- Scaler: {scaler_path}\n")
    f.write(f"- Histograma de puntuaciones: {plot_path}\n")

print(f"Log de entrenamiento guardado: {log_path}")
