import os, sys
import pandas as pd

'''
Script para concatenar los datasets de un directorio dada la ruta en línea de comandos.
Recibe en línea de comandos la ruta del directorio que contiene los ficheros .csv 
y el nombre del fichero de salida

Ejemplo de uso:
python concatenate_datasets.py "..\my_data_directory" "..\my_data_directory\concat_dataset.csv"
'''

#obtener argumentos
args = sys.argv

# Directorio que contiene los archivos CSV
directorio = args[1]

# Lista para almacenar los DataFrames de cada archivo CSV
dataframes = []

# Iterar sobre los archivos CSV en el directorio
for archivo in os.listdir(directorio):
    if archivo.endswith('.csv'):
        # Leer el archivo CSV y agregar el DataFrame a la lista
        ruta_archivo = os.path.join(directorio, archivo)
        df = pd.read_csv(ruta_archivo, low_memory=False)
        dataframes.append(df)

# Concatenar todos los DataFrames
df_final = pd.concat(dataframes, ignore_index=True)

# Guardar el DataFrame resultante en otro archivo CSV
ruta_salida = args[2]
df_final.to_csv(ruta_salida, index=False)  # index=False para no incluir el índice en el archivo CSV