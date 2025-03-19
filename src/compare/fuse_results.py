# Combinar la probabilidad de DM del modelo RandomForest (validado por ANN) con el score de anomalía del modelo OCSVM
# para obtener un score combinado que permita priorizar las fuentes más anómalas y con mayor probabilidad de ser materia oscura.
# fuse_results.py

import pandas as pd
import os
from sklearn.preprocessing import MinMaxScaler
import matplotlib.pyplot as plt

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

file_rf = os.path.join(project_root, 'outputs', 'predictions', 'unids_predictions_2025-03-19_12-56-02.csv')
df_rf = pd.read_csv(file_rf)

file_anomaly = os.path.join(project_root, 'outputs', 'anomalies', 'anomalies_ocsvm_astro.csv')
df_anomaly = pd.read_csv(file_anomaly)

# Unir DataFrames por 'number'
merged = pd.merge(df_rf, df_anomaly[['number', 'anomaly_score']], on='number')

# Normalizar anomaly_score para que esté entre 0 y 1 (más anómalo = mayor score)
scaler = MinMaxScaler()
merged['anomaly_score_norm'] = scaler.fit_transform(merged[['anomaly_score']])

merged['combined_score'] = 0.5 * merged['prob_DM'] + 0.5 * merged['anomaly_score_norm']

merged_sorted = merged.sort_values(by='combined_score', ascending=False)

output_path = os.path.join(project_root, 'outputs', 'results', 'unids_combined_ranking.csv')
os.makedirs(os.path.dirname(output_path), exist_ok=True)
merged_sorted.to_csv(output_path, index=False)

print(f"Ranking final guardado: {output_path}")
print(merged_sorted[['number', 'prob_DM', 'anomaly_score', 'anomaly_score_norm', 'combined_score']].head(10))

top10 = merged_sorted.head(10)
plt.figure(figsize=(10, 5))
plt.bar(top10['number'].astype(str), top10['combined_score'], color='#6a3d9a')
plt.xlabel('ID Fuente UNID')
plt.ylabel('Score Combinado')
plt.title('Top 10 Candidatas a Materia Oscura (Prob_DM + Anomalía)')
plt.tight_layout()
plt.savefig(os.path.join(project_root, 'outputs', 'results', 'top10_combined_score.png'))
plt.close()
print("Gráfico top 10 guardado.")
