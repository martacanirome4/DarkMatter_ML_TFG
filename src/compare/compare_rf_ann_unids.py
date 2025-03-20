# Comparación de las probabilidades de clasificación de las fuentes UNIDs
# Cruzamos el modelo de clasificación supervisada Random Forest con el modelo de Red Neuronal, ambos generan una probabilidad de que una fuente sea materia oscura.
# Se comparan las probabilidades de clasificación de las fuentes UNIDs por ambos modelos, y se guardan las fuentes clasificadas como materia oscura por ambos modelos con probabilidad mayor a 0.90.
# Se generan dos gráficos: un scatter plot de las probabilidades de ambos modelos, y un bar plot de las fuentes clasificadas como materia oscura por ambos modelos.
# Se guarda un archivo CSV con las fuentes clasificadas como materia oscura por ambos modelos.
# src/compare/compare_rf_ann_unids.py
# compare_rf_ann_unids.py
import os
import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === Cargar archivo ANN ===
def load_ann_data(ann_file):
    raw_data = np.genfromtxt(ann_file, dtype='str')
    data = np.asarray(raw_data[1:], dtype=float)  # Salta cabecera
    n_samples = data.shape[1] - 1

    columns = ['number'] + [f'prob_{i}' for i in range(n_samples)]
    df_full = pd.DataFrame(data, columns=columns)

    # Media por fuente y entre folds
    df_mean = df_full.groupby('number').mean().reset_index()
    df_mean['prob_ann'] = df_mean[[f'prob_{i}' for i in range(n_samples)]].mean(axis=1)

    return df_mean[['number', 'prob_ann']]

# === Buscar y cargar el CSV más reciente de predicciones RF ===
def load_latest_rf_predictions():
    pred_dir = os.path.join(project_root, 'outputs', 'predictions')
    csv_files = glob.glob(os.path.join(pred_dir, 'unids_predictions_*.csv'))

    if not csv_files:
        raise FileNotFoundError("No se encontraron archivos de predicciones RF en 'outputs/predictions'.")

    csv_files.sort(key=os.path.getmtime, reverse=True)
    latest_csv = csv_files[0]
    print(f"Archivo de predicción RF cargado: {latest_csv}")
    return pd.read_csv(latest_csv)

# === Comparar y filtrar candidatas comunes RF y ANN ===
def merge_and_filter(df_rf, df_ann, threshold=0.90):
    merged = pd.merge(df_rf, df_ann, on='number')
    both_high = merged[(merged['prob_DM'] > threshold) & (merged['prob_ann'] > threshold)]
    return merged, both_high

# === Visualizar scatter plot RF vs ANN ===
def plot_scatter(merged, both_high, output_dir, timestamp):
    plt.figure(figsize=(8, 6))
    plt.scatter(merged['prob_DM'], merged['prob_ann'], alpha=0.4, s=40, color='#6baed6', label='UNIDs')
    plt.scatter(both_high['prob_DM'], both_high['prob_ann'], color='#f03b20', s=70, edgecolors='black', label='Candidatas DM (>0.90)')
    plt.xlabel('Probabilidad Random Forest', fontsize=12)
    plt.ylabel('Probabilidad Red Neuronal', fontsize=12)
    plt.title('Comparación de Probabilidades RF vs ANN\nFuentes UNIDs', fontsize=14, weight='bold')
    plt.legend()
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.tight_layout()
    scatter_path = os.path.join(output_dir, f'prob_rf_vs_ann_scatter_{timestamp}.png')
    plt.savefig(scatter_path, dpi=300)
    plt.close()
    print(f"Scatter plot guardado: {scatter_path}")
    return scatter_path

# === Visualizar bar plot de candidatas comunes ===
def plot_bar_comparison(both_high, output_dir, timestamp):
    plt.figure(figsize=(10, 6))
    bar_width = 0.4
    indices = np.arange(len(both_high))
    plt.bar(indices, both_high['prob_DM'], bar_width, label='RF', color='#74c476')
    plt.bar(indices + bar_width, both_high['prob_ann'], bar_width, label='ANN', color='#fd8d3c')

    plt.xlabel('ID Fuente UNID', fontsize=12)
    plt.ylabel('Probabilidad', fontsize=12)
    plt.title('Comparación de Probabilidades en Candidatas DM', fontsize=14, weight='bold')
    plt.xticks(indices + bar_width / 2, both_high['number'].astype(int), rotation=45)
    plt.legend()
    plt.tight_layout()
    bar_path = os.path.join(output_dir, f'prob_candidatas_bar_{timestamp}.png')
    plt.savefig(bar_path, dpi=300)
    plt.close()
    print(f"Bar plot guardado: {bar_path}")
    return bar_path

# === Histograma comparativo RF vs ANN ===
def plot_histogram_comparison(merged, output_dir, timestamp):
    plt.figure(figsize=(10, 6))
    plt.hist(merged['prob_DM'], bins=20, alpha=0.6, label='RF', color='#3182bd', edgecolor='black')
    plt.hist(merged['prob_ann'], bins=20, alpha=0.6, label='ANN', color='#e6550d', edgecolor='black')
    plt.xlabel('Probabilidad')
    plt.ylabel('Cantidad de UNIDs')
    plt.title('Distribución Comparativa de Probabilidades\nRF vs ANN', fontsize=14, weight='bold')
    plt.legend()
    plt.tight_layout()
    hist_path = os.path.join(output_dir, f'hist_comparativo_rf_ann_{timestamp}.png')
    plt.savefig(hist_path, dpi=300)
    plt.close()
    print(f"Histograma comparativo guardado: {hist_path}")
    return hist_path

# === Guardar resultados comunes a CSV ===
def save_common_candidates(both_high, output_dir, timestamp):
    csv_path = os.path.join(output_dir, f'rf_ann_candidates_{timestamp}.csv')
    both_high.to_csv(csv_path, index=False)
    print(f"Candidatas RF-ANN guardadas en: {csv_path}")
    return csv_path 

# === Guardar log de comparación ===
def save_comparison_log(total_unids, total_candidates, scatter_path, bar_path, hist_path, csv_path, output_dir, timestamp):
    log_path = os.path.join(output_dir, f'comparison_log_{timestamp}.txt')
    with open(log_path, 'w') as f:
        f.write(f"=== Comparación RF vs ANN - UNIDs ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Total UNIDs procesadas: {total_unids}\n")
        f.write(f"Candidatas comunes RF y ANN (prob > 0.90): {total_candidates}\n\n")
        f.write("Gráficos generados:\n")
        f.write(f"- Scatter Plot: {scatter_path}\n")
        f.write(f"- Bar Plot: {bar_path}\n")
        f.write(f"- Histograma Comparativo: {hist_path}\n\n")
        f.write(f"CSV de candidatas: {csv_path}\n")
    print(f"Log guardado: {log_path}")

# === Función principal ===
def main():
    ann_file = os.path.join(project_root, 'notebooks', '4F_ANN', 'unids_DM_std_proba_check_repeated_kfold_rskf_4F_21.txt')

    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = os.path.join(project_root, 'outputs', 'compare')
    os.makedirs(output_dir, exist_ok=True)

    df_ann = load_ann_data(ann_file)
    df_rf = load_latest_rf_predictions()

    merged, both_high = merge_and_filter(df_rf, df_ann, threshold=0.90)

    print(f"Total UNIDs clasificadas como DM por ambos modelos (>0.90): {len(both_high)}")
    print(both_high[['number', 'prob_DM', 'prob_ann']].sort_values(by='prob_ann', ascending=False))

    scatter_path = plot_scatter(merged, both_high, output_dir, timestamp)
    bar_path = plot_bar_comparison(both_high, output_dir, timestamp)
    hist_path = plot_histogram_comparison(merged, output_dir, timestamp)
    csv_path = save_common_candidates(both_high, output_dir, timestamp)

    save_comparison_log(len(merged), len(both_high), scatter_path, bar_path, hist_path, csv_path, output_dir, timestamp)

    csv_path = save_common_candidates(both_high, output_dir, timestamp)


if __name__ == '__main__':
    main()
