# Fusionar los resultados de probabilidad DM de la red neuronal con los scores de anomalía del modelo OCSVM
# para priorizar las fuentes más anómalas y con mayor probabilidad de ser materia oscura.
# src/compare/fuse_ann_ocsvm_results.py
# fuse_ann_ocsvm_results.py
import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt
from datetime import datetime
import seaborn as sns

sns.set_style('whitegrid')
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

# === Cargar prob_ann desde archivo ANN original ===
ann_file = os.path.join(project_root, 'notebooks', '4F_ANN', 'unids_DM_std_proba_check_repeated_kfold_rskf_4F_21.txt')
raw_data = np.genfromtxt(ann_file, dtype='str')
data = np.asarray(raw_data[1:], dtype=float)
n_samples = data.shape[1] - 1
columns = ['number'] + [f'prob_{i}' for i in range(n_samples)]
df_ann = pd.DataFrame(data, columns=columns)
df_ann['prob_ann'] = df_ann[[f'prob_{i}' for i in range(n_samples)]].mean(axis=1)

# === Cargar archivo de anomalías más reciente ===
anomaly_dir = os.path.join(project_root, 'outputs', 'anomalies')
anomaly_csv_files = [f for f in os.listdir(anomaly_dir) if f.startswith('anomalies_ocsvm_astro_') and f.endswith('.csv')]
anomaly_csv_files.sort(key=lambda x: os.path.getmtime(os.path.join(anomaly_dir, x)), reverse=True)
latest_anomaly_file = os.path.join(anomaly_dir, anomaly_csv_files[0])
df_anomaly = pd.read_csv(latest_anomaly_file)
print(f"Archivo de anomalías cargado: {latest_anomaly_file}")

# Agrupar por 'number' y hacer la media de prob_ann si hay duplicados
df_ann_grouped = df_ann.groupby('number', as_index=False)['prob_ann'].mean()

# Fusionar con anomaly_score
merged = pd.merge(df_ann_grouped, df_anomaly[['number', 'anomaly_score']], on='number')

# === Normalizar anomaly_score ===
scaler = MinMaxScaler()
merged['anomaly_score_norm'] = scaler.fit_transform(merged[['anomaly_score']])

# === Calcular score combinado ===
merged['combined_score'] = 0.5 * merged['prob_ann'] + 0.5 * merged['anomaly_score_norm']
merged_sorted = merged.sort_values(by='combined_score', ascending=False)

results_dir = os.path.join(project_root, 'outputs', 'results')
os.makedirs(results_dir, exist_ok=True)

# === Guardar ranking completo ===
csv_path = os.path.join(results_dir, f'unids_combined_ann_ocsvm_{timestamp}.csv')
merged_sorted.to_csv(csv_path, index=False)
print(f"Ranking combinado guardado: {csv_path}")

top5 = merged_sorted.head(5)
top5_path = os.path.join(results_dir, f'top5_unids_ann_ocsvm_{timestamp}.csv')
top5.to_csv(top5_path, index=False)
print(f"Top 5 UNIDs guardado: {top5_path}")

# === Scatter plot ===
plt.figure(figsize=(10, 6))
scatter = plt.scatter(
    merged['prob_ann'],
    merged['anomaly_score_norm'],
    c=merged['combined_score'],
    cmap='plasma',
    s=50,
    edgecolor='k',
    alpha=0.7
)
cbar = plt.colorbar(scatter)
cbar.set_label('Score Combinado', fontsize=12)
plt.xlabel('Probabilidad DM - ANN', fontsize=12)
plt.ylabel('Anomaly Score Normalizado', fontsize=12)
plt.title(f'UNIDs: Probabilidad ANN vs Anomalía\n{timestamp}', fontsize=14, weight='bold')
plt.tight_layout()
scatter_path = os.path.join(results_dir, f'scatter_ann_vs_anomaly_{timestamp}.png')
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Scatter plot guardado: {scatter_path}")

# === Bar plot Top 10 ===
top10 = merged_sorted.head(10)
plt.figure(figsize=(10, 6))
colors = plt.cm.viridis(top10['combined_score'] / top10['combined_score'].max())
bars = plt.bar(top10['number'].astype(str), top10['combined_score'], color=colors, edgecolor='black')
plt.xlabel('ID Fuente UNID', fontsize=12)
plt.ylabel('Score Combinado', fontsize=12)
plt.title(f'Top 10 Candidatas DM (ANN + Anomalía)\n{timestamp}', fontsize=14, weight='bold')
plt.xticks(rotation=45)
plt.tight_layout()
bar_path = os.path.join(results_dir, f'top10_combined_ann_ocsvm_{timestamp}.png')
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Bar plot Top 10 guardado: {bar_path}")

# === Log resumen ===
log_path = os.path.join(results_dir, f'fuse_ann_ocsvm_log_{timestamp}.txt')
with open(log_path, 'w') as f:
    f.write(f"=== Fusión ANN + OCSVM ===\n")
    f.write(f"Timestamp: {timestamp}\n\n")
    f.write(f"Total UNIDs procesadas: {len(merged)}\n\n")

    f.write("Ranking combinado:\n")
    f.write(f"- Archivo completo: {csv_path}\n")
    f.write(f"- Top 5 CSV: {top5_path}\n\n")

    f.write("Gráficos generados:\n")
    f.write(f"- Scatter plot: {scatter_path}\n")
    f.write(f"- Bar plot Top 10: {bar_path}\n\n")

    f.write(f"Fuente de anomalías: {latest_anomaly_file}\n\n")

    f.write("=== Top 5 UNIDs (ANN + Anomalía) ===\n")
    f.write(top5[['number', 'prob_ann', 'anomaly_score_norm', 'combined_score']].to_string(index=False, float_format="%.4f"))
print(f"Log guardado con Top 5: {log_path}")
