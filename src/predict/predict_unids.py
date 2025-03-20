# Aplicamos el modelo de clasificación entre DM y fuentes astrofísicas a las UNIDs, y guardamos los resultados en un archivo CSV y un gráfico.
# src/predict/predict_unids.py 
# predict_unids.py
import os
import pandas as pd
import joblib
import matplotlib.pyplot as plt
from datetime import datetime
import glob

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === Cargar modelo entrenado ===
def load_model(model_filename):
    model_path = os.path.join(project_root, 'outputs', model_filename)
    clf = joblib.load(model_path)
    print(f"Modelo cargado: {model_path}")
    return clf

def load_latest_model():
    models_dir = os.path.join(project_root, 'outputs')
    model_files = glob.glob(os.path.join(models_dir, 'rf_model_*.joblib'))

    if not model_files:
        raise FileNotFoundError("No se encontraron modelos .joblib en la carpeta 'outputs'.")

    # Ordenar archivos por fecha de modificación (más reciente al final)
    model_files.sort(key=os.path.getmtime, reverse=True)
    latest_model_path = model_files[0]

    clf = joblib.load(latest_model_path)
    print(f"Modelo más reciente cargado: {latest_model_path}")
    return clf

# === Cargar datos UNIDs ===
def load_unids_data():
    unids_file = os.path.join(project_root, 'data', 'raw', 'unids_3F_beta_err_names.txt')
    df_unids = pd.read_csv(unids_file, sep='\s+')
    # Ajustar columna para que coincida con el modelo
    X_unids = df_unids[['E_peak', 'beta', 'sigma_det', 'beta_Rel']]
    X_unids = X_unids.rename(columns={'sigma_det': 'sigma'})
    return df_unids['number'], X_unids

# === Predecir probabilidades ===
def predict_probabilities(clf, numbers, X_unids):
    probs_dm = clf.predict_proba(X_unids)[:, 1]
    results_df = pd.DataFrame({
        'number': numbers,
        'prob_DM': probs_dm
    }).sort_values(by='prob_DM', ascending=False)
    return results_df

# === Guardar resultados a CSV ===
def save_results_csv(results_df, timestamp):
    output_dir = os.path.join(project_root, 'outputs', 'predictions')
    os.makedirs(output_dir, exist_ok=True)
    results_file = os.path.join(output_dir, f'unids_predictions_{timestamp}.csv')
    results_df.to_csv(results_file, index=False)
    print(f"Resultados guardados en: {results_file}")
    return output_dir, results_file

# === Graficar Top 20 UNIDs ===
def plot_top20(results_df, output_dir, timestamp):
    top20 = results_df.head(20)
    plt.figure(figsize=(10, 6))
    plt.bar(top20['number'].astype(str), top20['prob_DM'], color='slateblue')
    plt.xlabel('ID Fuente UNID')
    plt.ylabel('Probabilidad DM')
    plt.title(f'Top 20 UNIDs - Probabilidad Materia Oscura\n{timestamp}')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'unids_top20_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Gráfico Top 20 guardado: {plot_path}")

# === Graficar histograma de probabilidades ===
def plot_distribution(results_df, output_dir, timestamp):
    plt.figure(figsize=(10, 6))
    plt.hist(results_df['prob_DM'], bins=20, edgecolor='black', color='seagreen')
    plt.xlabel('Probabilidad de Materia Oscura')
    plt.ylabel('Cantidad de UNIDs')
    plt.title(f'Distribución de Probabilidades de Materia Oscura - UNIDs\n{timestamp}')
    plt.tight_layout()
    plot_path = os.path.join(output_dir, f'unids_prob_distribution_{timestamp}.png')
    plt.savefig(plot_path)
    plt.close()
    print(f"Histograma guardado: {plot_path}")

# === Contar UNIDs que superan umbrales ===
def count_thresholds(results_df):
    thresholds = [0.90, 0.95, 0.99]
    total = len(results_df)
    print(f"Total UNIDs: {total}")
    for thresh in thresholds:
        count = len(results_df[results_df['prob_DM'] > thresh])
        print(f"UNIDs con prob > {thresh:.2f}: {count}")

# === Función principal ===
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # model_filename = 'rf_model_2025-03-20_13-26-14.joblib' 
    # clf = load_model(model_filename)
    clf = load_latest_model()

    numbers, X_unids = load_unids_data()

    results_df = predict_probabilities(clf, numbers, X_unids)

    output_dir, results_file = save_results_csv(results_df, timestamp)

    plot_top20(results_df, output_dir, timestamp)
    plot_distribution(results_df, output_dir, timestamp)

    count_thresholds(results_df)

if __name__ == '__main__':
    main()
