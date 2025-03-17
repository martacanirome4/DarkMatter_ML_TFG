import sys
import os

# Añadir src/ al PYTHONPATH dinámicamente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data_processing.separate_data import separate_data

def main():
    print("🔍 Iniciando pipeline de procesamiento y modelado...\n")

    # 1. Separar datos
    separate_data()

    # 2. Aquí podrías llamar al modelo:
    # from models.one_class_svm import train_model
    # train_model()

    print("\n🎯 Pipeline completado exitosamente.")

if __name__ == "__main__":
    main()
