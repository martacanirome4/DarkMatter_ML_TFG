import os
import pandas as pd
import joblib
from datetime import datetime
import matplotlib.pyplot as plt

# Ruta raíz
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cargar modelo entrenado
model_path = os.path.join(project_root, 'outputs', 'rf_model_2025-03-19_12-36-44.joblib')
clf = joblib.load(model_path)
print(f"Modelo cargado: {model_path}")

# Cargar datos UNIDs
unids_file = os.path.join(project_root, 'data', 'raw', 'unids_3F_beta_err_names.txt')
df_unids = pd.read_csv(unids_file, sep='\s+')

# Procesar datos
X_unids = df_unids[['E_peak', 'beta', 'sigma_det', 'beta_Rel']]
X_unids = X_unids.rename(columns={'sigma_det': 'sigma'})  # Cambiar nombre para que coincida con el modelo
numbers = df_unids['number']

# Predecir probabilidades de DM
probs_dm = clf.predict_proba(X_unids)[:, 1]  # Probabilidad de ser materia oscura

# Crear dataframe resultados
results_df = pd.DataFrame({
    'number': numbers,
    'prob_DM': probs_dm
}).sort_values(by='prob_DM', ascending=False)

# Guardar resultados
output_dir = os.path.join(project_root, 'outputs', 'predictions')
os.makedirs(output_dir, exist_ok=True)
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
results_file = os.path.join(output_dir, f'unids_predictions_{timestamp}.csv')
results_df.to_csv(results_file, index=False)
print(f"Resultados guardados en: {results_file}")

# Graficar Top 20 candidatas
top20 = results_df.head(20)
plt.figure(figsize=(10, 6))
plt.bar(top20['number'].astype(str), top20['prob_DM'])
plt.xlabel('ID Fuente UNID')
plt.ylabel('Probabilidad DM')
plt.title(f'Top 20 UNIDs - Probabilidad Materia Oscura\n{timestamp}')
plt.xticks(rotation=45)
plot_path = os.path.join(output_dir, f'unids_top20_{timestamp}.png')
plt.tight_layout()
plt.savefig(plot_path)
plt.close()
print(f"Gráfico Top 20 guardado: {plot_path}")

# Leer archivo de resultados ya guardado
results_df = pd.read_csv('outputs/predictions/unids_predictions_2025-03-19_12-56-02.csv')

# Histograma
plt.figure(figsize=(10, 6))
plt.hist(results_df['prob_DM'], bins=20, edgecolor='black')
plt.xlabel('Probabilidad de Materia Oscura')
plt.ylabel('Cantidad de UNIDs')
plt.title(f'Distribución de Probabilidades de Materia Oscura - UNIDs\n{timestamp}')
plt.tight_layout()

# Guardar gráfico
dist_plot_path = f'outputs/predictions/unids_prob_distribution_{timestamp}.png'
plt.savefig(dist_plot_path)
plt.close()
print(f"Histograma guardado: {dist_plot_path}")

# Contar cuántas UNIDs superan ciertos umbrales
umbral_90 = results_df[results_df['prob_DM'] > 0.9]
umbral_95 = results_df[results_df['prob_DM'] > 0.95]
umbral_99 = results_df[results_df['prob_DM'] > 0.99]

print(f"Total UNIDs: {len(results_df)}")
print(f"UNIDs con prob > 0.90: {len(umbral_90)}")
print(f"UNIDs con prob > 0.95: {len(umbral_95)}")
print(f"UNIDs con prob > 0.99: {len(umbral_99)}")

