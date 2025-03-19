import os
import pandas as pd

def separate_data(input_filename='XY_bal_log_Rel.txt',
                  input_dir='data/raw',
                  output_dir='data/processed/XY_bal_log_Rel'):
    """
    Separa los datos en astro y DM a partir de un archivo con etiquetas 0.0 y 1.0,
    y guarda los resultados en carpetas separadas.

    Parámetros:
    - input_filename: nombre del archivo de entrada.
    - input_dir: carpeta donde se encuentra el archivo.
    - output_dir: carpeta raíz donde guardar los datos separados.
    """
    # Ruta absoluta a la raíz del proyecto
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
    
    # Construir rutas completas
    input_path = os.path.join(project_root, input_dir, input_filename)
    astro_dir = os.path.join(project_root, output_dir, 'astro')
    dm_dir = os.path.join(project_root, output_dir, 'DM')
    
    # Verificar que el archivo existe
    if not os.path.isfile(input_path):
        raise FileNotFoundError(f"No se encontró el archivo: {input_path}")
    
    # Crear carpetas de salida si no existen
    os.makedirs(astro_dir, exist_ok=True)
    os.makedirs(dm_dir, exist_ok=True)

    # Leer datos
    data = pd.read_csv(input_path, sep='\s+', header=0)
    
    # Renombrar columna de etiquetas
    data = data.rename(columns={'0,1=astro,DM': 'label'})

    # Separar datos
    astro = data[data['label'] == 0.0]
    dm = data[data['label'] == 1.0]

    # Rutas de salida
    astro_file = os.path.join(astro_dir, 'XY_bal_log_Rel_astro.txt')
    dm_file = os.path.join(dm_dir, 'XY_bal_log_Rel_DM.txt')

    # Guardar archivos
    astro.to_csv(astro_file, sep=' ', index=False, header=False)
    dm.to_csv(dm_file, sep=' ', index=False, header=False)

    print(f"✅ Datos astro guardados en: {astro_file}")
    print(f"✅ Datos DM guardados en: {dm_file}")

# Permite ejecutar este archivo directamente para probarlo
if __name__ == "__main__":
    separate_data()
