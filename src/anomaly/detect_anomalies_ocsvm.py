import pandas as pd
import os
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
file_path = os.path.join(project_root, 'data', 'raw', 'XY_bal_log_Rel.txt')

df = pd.read_csv(file_path, sep='\s+')
df_astro = df[df['0,1=astro,DM'] == 0.0]
X_astro = df_astro[['E_peak', 'beta', 'sigma', 'beta_Rel']].copy()

print(f"Total fuentes ASTRO: {len(X_astro)}")
print(X_astro.head())

# Normalizar datos astro
scaler = StandardScaler()
X_astro_scaled = scaler.fit_transform(X_astro)

# Entrenar modelo One-Class SVM
ocsvm = OneClassSVM(kernel='rbf', gamma='scale', nu=0.05)
ocsvm.fit(X_astro_scaled)

print("Modelo One-Class SVM entrenado con datos ASTRO")

# Guardar modelo y scaler
model_dir = os.path.join(project_root, 'outputs', 'models')
os.makedirs(model_dir, exist_ok=True)

joblib.dump(ocsvm, os.path.join(model_dir, 'ocsvm_astro_model.joblib'))
joblib.dump(scaler, os.path.join(model_dir, 'scaler_astro.joblib'))
print("Modelo y scaler guardados.")