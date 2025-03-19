# Clasificación con Random Forest, aprende con datos astro y DM a diferenciar entre fuentes de materia oscura y otras fuentes astrofísicas
# classifier_rf.py

import os
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, roc_curve, auc
from datetime import datetime

project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))

def load_data(filepath):
    df_file = os.path.join(project_root, 'data', 'raw', 'XY_bal_log_Rel.txt')

    if not os.path.exists(df_file):
        print(f"❌ Archivo no encontrado: {df_file}")
        return None, None
    else:
        print(f"✅ Cargando archivo: {df_file}")
        df = pd.read_csv(df_file, sep='\s+')

    X = df[['E_peak', 'beta', 'sigma', 'beta_Rel']]
    y = df['0,1=astro,DM']
    return X, y


def train_classifier(X_train, y_train):
    clf = RandomForestClassifier(n_estimators=100, random_state=42)
    clf.fit(X_train, y_train)
    return clf


def evaluate_model(clf, X_test, y_test):
    y_pred = clf.predict(X_test)
    y_prob = clf.predict_proba(X_test)[:, 1]
    print("Classification Report:")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    roc_auc = roc_auc_score(y_test, y_prob)
    print(f"ROC AUC Score: {roc_auc:.4f}")
    return y_pred, y_prob, roc_auc


def main():
    # Cargar datos y dividir en entrenamiento y test
    filepath = '../../raw/data/XY_bal_log_Rel.txt'
    X, y = load_data(filepath)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y)
    # Entrenar y evaluar modelo
    clf = train_classifier(X_train, y_train)
    y_pred, y_prob, roc_auc = evaluate_model(clf, X_test, y_test)

    # Guardar resultados -------------------------------------------------------
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    output_dir = os.path.join(project_root, 'outputs')
    images_dir = os.path.join(output_dir, 'images')
    logs_dir = os.path.join(output_dir, 'logs')

    conf_matrix_path = os.path.join(images_dir, f'confusion_matrix_rf_{timestamp}.png')
    roc_curve_path = os.path.join(images_dir, f'roc_curve_rf_{timestamp}.png')
    feature_importance_path = os.path.join(images_dir, f'feature_importance_rf_{timestamp}.png')
    log_path = os.path.join(logs_dir, f'metrics_log_rf_{timestamp}.txt')

    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(logs_dir, exist_ok=True)


    # Matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Astro', 'DM'], yticklabels=['Astro', 'DM'])
    plt.xlabel('Predicho')
    plt.ylabel('Real')
    plt.title(f'Matriz de Confusión - Random Forest\n{timestamp}')
    plt.savefig(conf_matrix_path)
    plt.close()


    # Curva ROC
    fpr, tpr, thresholds = roc_curve(y_test, y_prob)
    roc_auc = auc(fpr, tpr)

    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title(f'Curva ROC - Random Forest\n{timestamp}')
    plt.legend(loc="lower right")
    plt.savefig(roc_curve_path)
    plt.close()


    # Guardar importancia de características
    importances = clf.feature_importances_
    feature_names = X.columns

    plt.figure()
    plt.barh(feature_names, importances)
    plt.xlabel('Importancia')
    plt.title(f'Importancia de Características - Random Forest\n{timestamp}')
    plt.savefig(feature_importance_path)
    plt.close()


    # Guardar métricas en log .txt
    with open(log_path, 'w') as f:
        f.write(f"=== Random Forest - Clasificación Astro vs DM ===\n")
        f.write(f"Timestamp: {timestamp}\n\n")
        f.write("Classification Report:\n")
        f.write(classification_report(y_test, y_pred))
        f.write("\nConfusion Matrix:\n")
        f.write(str(cm))
        f.write(f"\n\nROC AUC Score: {roc_auc:.4f}\n")
        f.write("\nGráficos generados:\n")
        f.write(f"- Matriz de Confusión: {conf_matrix_path}\n")
        f.write(f"- Curva ROC: {roc_curve_path}\n")
        f.write(f"- Importancia Variables: {feature_importance_path}\n")


    # Guardar modelo entrenado
    model_path = os.path.join(output_dir, f'rf_model_{timestamp}.joblib')
    joblib.dump(clf, model_path)
    print(f"✅ Modelo guardado en: {model_path}")

if __name__ == '__main__':
    main()
