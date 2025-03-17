import sys
import os

# Añadir src/ al PYTHONPATH dinámicamente
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from data_processing.separate_data import separate_data
from models.one_class_svm import train_model
from models.apply_oneclasssvm_unids import apply_oneclasssvm_to_unids

def main():

    print("🔍 Iniciando pipeline de procesamiento y modelado...\n")
    separate_data()

    train_model(save_plots=True, show_plots=False)
    print("\n🎯 Pipeline completo.")

    apply_oneclasssvm_to_unids()
    print("\n🎯 Pipeline completo.")
 
if __name__ == "__main__":
    main()
