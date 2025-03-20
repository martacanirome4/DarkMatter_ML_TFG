# Aplicar el modelo entrenado de One-Class SVM a los datos de test en busca de anomalías detect_anomalies_ocsvm.py
# Se busca detectar anomalías en las fuentes astrofísicas (ASTRO) con el modelo entrenado de One-Class SVM.
# Se cargan los datos de test, se normalizan y se predicen las anomalías.
# Se guardan los resultados en un archivo CSV y se genera un histograma de los scores de anomalía.
# src/anomaly/predict_anomalies_ocsvm.py
# predict_anomalies_ocsvm.py
import pandas as pd
import os
import glob
from sklearn.preprocessing import StandardScaler
from sklearn.svm import OneClassSVM
import joblib
import matplotlib.pyplot as plt
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === Cargar el modelo y scaler más reciente ===
def load_latest_ocsvm_and_scaler():
    model_dir = os.path.join(project_root, 'outputs', 'models')
    ocsvm_files = glob.glob(os.path.join(model_dir, 'ocsvm_astro_model_*.joblib'))
    scaler_files = glob.glob(os.path.join(model_dir, 'scaler_astro_*.joblib'))

    if not ocsvm_files or not scaler_files:
        raise FileNotFoundError("No se encontraron modelos o scalers OCSVM en 'outputs/models/'")

    # Ordenar por fecha de modificación
    ocsvm_files.sort(key=os.path.getmtime, reverse=True)
    scaler_files.sort(key=os.path.getmtime, reverse=True)

    latest_model_path = ocsvm_files[0]
    latest_scaler_path = scaler_files[0]

    ocsvm = joblib.load(latest_model_path)
    scaler = joblib.load(latest_scaler_path)

    print(f"Modelo cargado: {latest_model_path}")
    print(f"Scaler cargado: {latest_scaler_path}")

    return ocsvm, scaler, latest_model_path, latest_scaler_path

# === Guardar log resumen de detección ===
def save_detection_log(total, n_anomalies, csv_path, plot_path, model_path, scaler_path, log_dir, timestamp):
    log_path = os.path.join(log_dir, f'anomaly_detection_log_{timestamp}.txt')
    with open(log_path, 'w') as f:
        f.write(f"=== Detección de Anomalías - UNIDs ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write(f"Total fuentes UNIDs analizadas: {total}\n")
        f.write(f"Anomalías detectadas: {n_anomalies}\n")
        f.write(f"Porcentaje anomalías: {100 * n_anomalies / total:.2f}%\n\n")
        f.write("Archivos generados:\n")
        f.write(f"- Resultados CSV: {csv_path}\n")
        f.write(f"- Histograma de scores: {plot_path}\n\n")
        f.write("Modelo usado:\n")
        f.write(f"- Modelo OCSVM: {model_path}\n")
        f.write(f"- Scaler: {scaler_path}\n")
    print(f"Log guardado: {log_path}")

# === Función principal ===
def main():
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    file_path = os.path.join(project_root, 'data', 'raw', 'unids_3F_beta_err_names.txt')
    df_test = pd.read_csv(file_path, sep='\s+')
    df_test.rename(columns={'sigma_det': 'sigma'}, inplace=True)
    X_test = df_test[['E_peak', 'beta', 'sigma', 'beta_Rel']].copy()

    ocsvm, scaler, model_path, scaler_path = load_latest_ocsvm_and_scaler()

    # === Normalizar datos UNIDs ===
    X_test_scaled = scaler.transform(X_test)

    # === Predecir anomalías y obtener scores ===
    y_pred = ocsvm.predict(X_test_scaled)
    anomaly_scores = ocsvm.decision_function(X_test_scaled)

    df_test['anomaly'] = y_pred  # -1 = anomalía, 1 = normal
    df_test['anomaly_score'] = anomaly_scores

    print("Conteo de anomalías detectadas:")
    print(df_test['anomaly'].value_counts())

    anomaly_dir = os.path.join(project_root, 'outputs', 'anomalies')
    os.makedirs(anomaly_dir, exist_ok=True)

    csv_path = os.path.join(anomaly_dir, f'anomalies_ocsvm_astro_{timestamp}.csv')
    df_test.to_csv(csv_path, index=False)
    print(f"Resultados guardados en: {csv_path}")

    # === Histograma de scores ===
    plt.figure(figsize=(10, 6))
    plt.hist(df_test['anomaly_score'], bins=30, edgecolor='black', color='salmon')
    plt.xlabel('Score de Anomalía (OCSVM)')
    plt.ylabel('Cantidad de UNIDs')
    plt.title(f'Distribución de Scores de Anomalía - UNIDs\n{timestamp}', fontsize=14, weight='bold')
    plt.tight_layout()
    plot_path = os.path.join(anomaly_dir, f'anomaly_score_distribution_{timestamp}.png')
    plt.savefig(plot_path, dpi=300)
    plt.close()
    print(f"Histograma guardado: {plot_path}")

    # === Guardar log resumen ===
    save_detection_log(
        total=len(df_test),
        n_anomalies=(df_test['anomaly'] == -1).sum(),
        csv_path=csv_path,
        plot_path=plot_path,
        model_path=model_path,
        scaler_path=scaler_path,
        log_dir=anomaly_dir,
        timestamp=timestamp
    )

if __name__ == '__main__':
    main()
