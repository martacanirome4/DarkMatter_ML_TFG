# FILE: src/models/apply_oneclasssvm_unids.py a
import os
import pandas as pd
import numpy as np
from sklearn.svm import OneClassSVM
from sklearn.preprocessing import StandardScaler

def apply_oneclasssvm_to_unids():
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    astro_file = os.path.join(project_root, 'data', 'processed', 'XY_bal_log_Rel', 'astro', 'XY_bal_log_Rel_astro.txt')
    unids_file = os.path.join(project_root, 'data', 'raw', 'unids_3F_beta_err_names.txt')

    # Leer datos
    astro_df = pd.read_csv(astro_file, sep=' ', header=None)
    unids_df = pd.read_csv(unids_file, sep='\s+', header=0)

    # Features unIDs (mismos que Astro)
    astro_features = astro_df.iloc[:, :-1].values  # 4 features
    unids_features = unids_df[['E_peak', 'beta', 'sigma_det', 'beta_Rel']].values

    # Escalar los datos
    scaler = StandardScaler()
    astro_scaled = scaler.fit_transform(astro_features)
    unids_scaled = scaler.transform(unids_features)

    # Entrenar OneClassSVM con Astro
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    model.fit(astro_scaled)

    # Predecir en unIDs
    unids_pred = model.predict(unids_scaled)  # 1 = normal, -1 = anomalía
    unids_df['prediction'] = unids_pred

    # Contar resultados
    print(unids_df['prediction'].value_counts())
    print("\nEjemplo de resultados:")
    print(unids_df.head())

    # Guardar anomalías
    anomalies_df = unids_df[unids_df['prediction'] == -1]
    output_path = os.path.join(project_root, 'outputs', 'unids_anomalies.csv')
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    anomalies_df.to_csv(output_path, index=False)
    print(f"\n✅ Fuentes anómalas guardadas en: {output_path}")

    print(f"\n✅ Proceso completado. {len(anomalies_df)} fuentes marcadas como anómalas.")

if __name__ == "__main__":
    apply_oneclasssvm_to_unids()
