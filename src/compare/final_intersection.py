# Buscar intersección entre top 10 fusionadas y candidatas RF-ANN
# Guardar archivo final etiquetando el origen de cada fuente
# final_intersection.py

import pandas as pd
import os
import matplotlib.pyplot as plt


project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# Cargar top 10 candidatas del ranking combinado
combined_file = os.path.join(project_root, 'outputs', 'results', 'unids_combined_ranking.csv')
df_combined = pd.read_csv(combined_file)
top_combined = df_combined.head(10)

# Cargar candidatas RF-ANN
rf_ann_file = os.path.join(project_root, 'outputs', 'compare', 'rf_ann_candidates.csv')
df_rf_ann = pd.read_csv(rf_ann_file)

# --- 1. Fusionar por 'number', asegurando que prob_DM no se duplique ---
merged = pd.merge(
    top_combined,  # contiene prob_DM (fusion)
    df_rf_ann[['number', 'prob_ann']],  # solo traemos prob_ann
    on='number', how='outer', indicator=True
)

# --- 2. Etiqueta de origen ---
merged['source'] = merged['_merge'].map({
    'both': 'Fusion + RF-ANN',
    'left_only': 'Fusion Only',
    'right_only': 'RF-ANN Only'
})

# --- 3. Ordenar por score y guardar ---
merged_sorted = merged.sort_values(by='combined_score', ascending=False)
final_path = os.path.join(project_root, 'outputs', 'results', 'final_candidates.csv')
merged_sorted[['number', 'prob_DM', 'prob_ann', 'combined_score', 'source']].to_csv(final_path, index=False)

print(merged_sorted[['number', 'prob_DM', 'combined_score', 'source']].head(15))
print(f"Archivo final de candidatas guardado: {final_path}")

# --- 4. Gráfico de barras ---
plt.figure(figsize=(10, 6))
bar_width = 0.25
index = range(len(merged_sorted))
plt.bar(index, merged_sorted['prob_DM'], bar_width, label='RF', color='#74c476')
plt.bar([i + bar_width for i in index], merged_sorted['prob_ann'], bar_width, label='ANN', color='#fd8d3c')
plt.bar([i + 2 * bar_width for i in index], merged_sorted['combined_score'], bar_width, label='Combined', color='#6a3d9a')
plt.xlabel('ID Fuente UNID', fontsize=12)
plt.ylabel('Score', fontsize=12)
plt.title('Comparación de Scores en Candidatas DM', fontsize=14, weight='bold')
plt.xticks([i + bar_width for i in index], merged_sorted['number'].astype(str))
plt.legend()
plt.tight_layout()
bar_path = os.path.join(project_root, 'outputs', 'results', 'final_candidates_bar.png')
plt.savefig(bar_path, dpi=300)
plt.close()
print(f"Gráfico de barras guardado: {bar_path}")