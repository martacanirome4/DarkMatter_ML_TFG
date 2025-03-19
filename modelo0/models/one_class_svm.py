# FILE: src/models/one_class_svm.py
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.svm import OneClassSVM
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix

def train_model(save_plots=True, show_plots=False):

    import datetime

    timestamp = datetime.datetime.now().strftime('%Y-%m-%d_%H-%M')

    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

    astro_file = os.path.join(project_root, 'data', 'processed', 'XY_bal_log_Rel', 'astro', 'XY_bal_log_Rel_astro.txt')
    dm_file = os.path.join(project_root, 'data', 'processed', 'XY_bal_log_Rel', 'DM', 'XY_bal_log_Rel_DM.txt')

    astro_data = pd.read_csv(astro_file, sep=' ', header=None)
    dm_data = pd.read_csv(dm_file, sep=' ', header=None)

    # Features
    # leer todas las columnas excepto la última
    X_astro = astro_data.iloc[:, :-1].values # -1 para excluir la última columna
    X_dm = dm_data.iloc[:, :-1].values

    # Etiquetas
    # etiquetas positivas para astro, negativas para DM
    y_astro = [1] * len(X_astro)
    y_dm = [-1] * len(X_dm)

    # Combinar astro y DM
    X_all = np.vstack((X_astro, X_dm))
    y_all = np.array(y_astro + y_dm)

    # Dividir en entrenamiento y test
    _, X_test, _, y_test = train_test_split(X_all, y_all, test_size=0.3, random_state=42)

    # Entrenar modelo: parámetros por defecto: kernel='rbf', gamma='auto', nu=0.1
    # parámetros ajustables: kernel, gamma, nu
    model = OneClassSVM(kernel='rbf', gamma='auto', nu=0.1)
    model.fit(X_astro)

    # Evaluar modelo: lo que hace es predecir si los datos son normales o anómalos
    y_pred = model.predict(X_test)

    print("🔍 Confusion Matrix:")
    # La matriz de confusión es una tabla con 4 celdas:
    # Verdaderos Positivos (VP): predijo correctamente que era normal
    # Verdaderos Negativos (VN): predijo correctamente que era anómalo
    # Falsos Positivos (FP): predijo incorrectamente que era normal
    # Falsos Negativos (FN): predijo incorrectamente que era anómalo
    print(confusion_matrix(y_test, y_pred))

    print("\n📊 Classification Report:")
    # El classification report muestra métricas como precisión, recall y f1-score
    print(classification_report(y_test, y_pred))

    # Crear carpeta logs
    logs_dir = os.path.join(project_root, 'outputs', 'logs')
    os.makedirs(logs_dir, exist_ok=True)

    # Guardar métricas en .txt
    metrics_path = os.path.join(logs_dir, f'{timestamp}_metrics.txt')
    with open(metrics_path, 'w') as f:
        f.write("Confusion Matrix:\n")
        f.write(str(confusion_matrix(y_test, y_pred)))
        f.write("\n\nClassification Report:\n")
        f.write(classification_report(y_test, y_pred))

    print(f"📝 Métricas guardadas en: {metrics_path}")

    # Crear carpeta para guardar figuras
    figures_dir = os.path.join(project_root, 'outputs', 'figures')
    os.makedirs(figures_dir, exist_ok=True)

    # ------ GRAFICO 1: Datos reales ------
    X_vis = X_all[:, :2]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_all, palette={1: 'blue', -1: 'red'}, alpha=0.6)
    plt.title("Distribución de datos reales (Astro=azul, DM=rojo)")
    plt.xlabel('Log(E_peak)')
    plt.ylabel('Log(beta)')
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(figures_dir, f'{timestamp}_datos_reales.png'))
    if show_plots:
        plt.show()
    plt.close()

    # ------ GRAFICO 2: Frontera de decisión ------
    xx, yy = np.meshgrid(np.linspace(X_vis[:, 0].min(), X_vis[:, 0].max(), 500),
                         np.linspace(X_vis[:, 1].min(), X_vis[:, 1].max(), 500))

    # Necesitamos 4 features: completamos los otros 2 con valores promedio
    X_dummy = np.c_[xx.ravel(), yy.ravel()]
    mean_feature3 = X_astro[:, 2].mean()
    mean_feature4 = X_astro[:, 3].mean()

    # Añadir columnas extra
    X_full = np.c_[X_dummy, np.full_like(xx.ravel(), mean_feature3), np.full_like(xx.ravel(), mean_feature4)]
    Z = model.predict(X_full)

    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(10, 6))
    plt.contourf(xx, yy, Z, cmap=plt.cm.coolwarm, alpha=0.4)
    sns.scatterplot(x=X_vis[:, 0], y=X_vis[:, 1], hue=y_all, palette={1: 'blue', -1: 'red'}, alpha=0.6)
    plt.title("Frontera del modelo OneClassSVM")
    plt.xlabel('Log(E_peak)')
    plt.ylabel('Log(beta)')
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(figures_dir, f'{timestamp}_frontera_svm.png'))
    if show_plots:
        plt.show()
    plt.close()

    # ------ GRAFICO 3: Predicciones ------

    # ------- DOCUMENTACIÓN: Visualización vs Entrenamiento -------
    # Los datos tienen 4 features reales:
    # 1. Log(E_peak), 2. Log(beta), 3. Log(sigma), 4. Log(beta_rel)

    # Para visualizar en 2D, proyectamos solo los 2 primeros (E_peak y beta).
    # Sin embargo, el modelo espera los 4 features como input.
    # Para graficar la frontera de decisión, completamos los otros 2 (sigma y beta_rel)
    # con su valor promedio en los datos de entrenamiento (Astro).
    # Esto nos permite "fingir" que estamos en 4D mientras visualizamos en 2D.

    X_vis_test = X_test[:, :2]
    plt.figure(figsize=(10, 6))
    sns.scatterplot(x=X_vis_test[:, 0], y=X_vis_test[:, 1], hue=y_pred, palette={1: 'green', -1: 'orange'}, alpha=0.6)
    plt.title("Predicción del modelo (verde=normal, naranja=anómalo)")
    plt.xlabel('Log(E_peak)')
    plt.ylabel('Log(beta)')
    plt.grid(True)
    if save_plots:
        plt.savefig(os.path.join(figures_dir, f'{timestamp}_predicciones.png'))
    if show_plots:
        plt.show()
    plt.close()

    print(f"📁 Figuras guardadas en: {figures_dir}")


if __name__ == "__main__":
    train_model()
