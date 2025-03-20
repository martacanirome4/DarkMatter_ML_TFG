# Clasificación con Random Forest, aprende con datos astro y DM a diferenciar entre fuentes de materia oscura y otras fuentes astrofísicas
# classifier_rf.py
import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve
from datetime import datetime

# === Define la raíz del proyecto ===
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

# === Carga de datos desde archivo .txt ===
def load_data():
    df_file = os.path.join(project_root, 'data', 'raw', 'XY_bal_log_Rel.txt')

    if not os.path.exists(df_file):
        print(f"Archivo no encontrado: {df_file}")
        return None, None
    else:
        print(f"Cargando archivo: {df_file}")
        # Lee el archivo separado por espacios
        df = pd.read_csv(df_file, sep='\s+')

    # Selección de características (X) y etiquetas (y)
    X = df[['E_peak', 'beta', 'sigma', 'beta_Rel']]
    y = df['0,1=astro,DM']
    return X, y

# === Entrenamiento del modelo Random Forest ===
def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf

# === Evaluación del modelo: métricas y predicciones ===
def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]  # Probabilidades clase positiva (DM)
    roc_auc = roc_auc_score(y_test, y_prob)

    # Reporte de clasificación y matriz de confusión
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    print(f"ROC AUC Score: {roc_auc:.4f}")

    return y_pred, y_prob, roc_auc

# === Visualización: matriz de confusión ===
def plot_confusion_matrix(cm, timestamp, save_path):
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Astro', 'DM'], yticklabels=['Astro', 'DM'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - Random Forest\n{timestamp}')
    plt.savefig(save_path)
    plt.close()

# === Visualización: curva ROC ===
def plot_roc_curve(fpr, tpr, roc_auc, timestamp, save_path):
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')  # Línea aleatoria
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - Random Forest\n{timestamp}')
    plt.legend(loc="lower right")
    plt.savefig(save_path)
    plt.close()

# === Visualización: importancia de características ===
def plot_feature_importance(importances, feature_names, timestamp, save_path):
    plt.figure()
    plt.barh(feature_names, importances)
    plt.xlabel('Importancia')
    plt.title(f'Importancia de Características - Random Forest\n{timestamp}')
    plt.savefig(save_path)
    plt.close()

# === Guardado de métricas y rutas de gráficos en archivo .txt ===
def save_metrics_log(cm, report, roc_auc, conf_matrix_path, roc_curve_path, feature_importance_path, log_path, timestamp):
    with open(log_path, 'w') as f:
        f.write(f"=== Random Forest - Clasificación Astro vs DM ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Classification Report:\n")
        f.write(report)
        f.write("\nConfusion Matrix:\n")
        f.write(pd.DataFrame(cm).__str__())  # Formato legible de matriz
        f.write(f"\n\nROC AUC Score: {roc_auc:.4f}\n")
        f.write("\nGráficos generados:\n")
        f.write(f"- Matriz de Confusión: {conf_matrix_path}\n")
        f.write(f"- Curva ROC: {roc_curve_path}\n")
        f.write(f"- Importancia Variables: {feature_importance_path}\n")

# === Función principal ===
def main():
    # 1. Carga de datos
    X, y = load_data()
    if X is None or y is None:
        return

    # 2. División de datos en entrenamiento y prueba (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)

    # 3. Entrenar modelo y evaluar
    clf = train_classifier(X_train, y_train)
    y_pred, y_prob, roc_auc = evaluate_model(clf, X_test, y_test)

    # 4. Preparar directorios y nombres de archivo con timestamp
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_dir = os.path.join(project_root, 'outputs')
    images_dir = os.path.join(output_dir, 'images')
    logs_dir = os.path.join(output_dir, 'logs')

    for directory in [images_dir, logs_dir]:
        os.makedirs(directory, exist_ok=True)

    conf_matrix_path = os.path.join(images_dir, f'confusion_matrix_rf_{timestamp}.png')
    roc_curve_path = os.path.join(images_dir, f'roc_curve_rf_{timestamp}.png')
    feature_importance_path = os.path.join(images_dir, f'feature_importance_rf_{timestamp}.png')
    log_path = os.path.join(logs_dir, f'metrics_log_rf_{timestamp}.txt')

    # 5. Generar gráficos
    cm = confusion_matrix(y_test, y_pred)
    fpr, tpr, _ = roc_curve(y_test, y_prob)
    importances = clf.feature_importances_
    feature_names = X.columns

    plot_confusion_matrix(cm, timestamp, conf_matrix_path)
    plot_roc_curve(fpr, tpr, roc_auc, timestamp, roc_curve_path)
    plot_feature_importance(importances, feature_names, timestamp, feature_importance_path)

    # 6. Guardar métricas y rutas de gráficos
    report = classification_report(y_test, y_pred)
    save_metrics_log(cm, report, roc_auc, conf_matrix_path, roc_curve_path, feature_importance_path, log_path, timestamp)

    # 7. Guardar modelo entrenado
    model_path = os.path.join(output_dir, f'rf_model_{timestamp}.joblib')
    joblib.dump(clf, model_path)
    print(f"Modelo guardado en: {model_path}")

if __name__ == '__main__':
    main()
