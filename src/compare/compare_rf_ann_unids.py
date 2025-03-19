import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

# Ruta al archivo ANN
file_ann = 'notebooks/4F_ANN/unids_DM_std_proba_check_repeated_kfold_rskf_4F_21.txt'
unids_ann_raw = np.genfromtxt(file_ann, dtype='str')
unids_ann_data = np.asarray(unids_ann_raw[1:], dtype=float)

# Preparar columnas
n_samples = unids_ann_data.shape[1] - 1
columns = ['number'] + [f'prob_{i}' for i in range(n_samples)]
df_ann_full = pd.DataFrame(unids_ann_data, columns=columns)

# Agrupar y calcular media por UNID
df_ann_mean = df_ann_full.groupby('number').mean().reset_index()
df_ann_mean['prob_ann'] = df_ann_mean[[f'prob_{i}' for i in range(n_samples)]].mean(axis=1)
df_ann_final = df_ann_mean[['number', 'prob_ann']]

# Cargar tus resultados
file_rf = 'outputs/predictions/unids_predictions_2025-03-19_12-56-02.csv'
df_rf = pd.read_csv(file_rf)

# Unir ambos DataFrames
merged = pd.merge(df_rf, df_ann_final, on='number')

# Filtrar por probabilidad alta en ambos modelos
both_high = merged[(merged['prob_DM'] > 0.90) & (merged['prob_ann'] > 0.90)]

print(f"Total UNIDs clasificadas como DM por ambos modelos (>0.90): {len(both_high)}")
print(both_high[['number', 'prob_DM', 'prob_ann']].sort_values(by='prob_ann', ascending=False))

# --- 1. Scatter Plot ---
plt.figure(figsize=(8, 6))
plt.scatter(merged['prob_DM'], merged['prob_ann'], alpha=0.4, s=40, color='#6baed6', label='UNIDs')
plt.scatter(both_high['prob_DM'], both_high['prob_ann'], color='#f03b20', s=70, edgecolors='black', label='Candidatas DM (>0.90)')
plt.xlabel('Probabilidad Random Forest', fontsize=12)
plt.ylabel('Probabilidad Red Neuronal', fontsize=12)
plt.title('Comparación de Probabilidades RF vs ANN\nFuentes UNIDs', fontsize=14, weight='bold')
plt.legend()
plt.grid(True, linestyle='--', alpha=0.5)
plt.tight_layout()
scatter_path = 'outputs/compare/prob_rf_vs_ann_scatter.png'
os.makedirs('outputs/compare', exist_ok=True)
plt.savefig(scatter_path, dpi=300)
plt.close()
print(f"Scatter plot guardado: {scatter_path}")

# --- 2. Bar Plot Top Candidatas ---
plt.figure(figsize=(8, 5))
bar_width = 0.35
index = range(len(both_high))
plt.bar(index, both_high['prob_DM'], bar_width, label='RF', color='#74c476')
plt.bar([i + bar_width for i in index], both_high['prob_ann'], bar_width, label='ANN', color='#fd8d3c')
plt.xlabel('ID Fuente UNID', fontsize=12)
plt.ylabel('Probabilidad', fontsize=12)
plt.title('Comparación de Probabilidades en Candidatas DM', fontsize=14, weight='bold')
plt.xticks([i + bar_width / 2 for i in index], both_high['number'].astype(str))
plt.legend()
plt.tight_layout()
bar_path = f'outputs/compare/prob_candidatas_bar.png'
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Bar plot guardado: {bar_path}")
