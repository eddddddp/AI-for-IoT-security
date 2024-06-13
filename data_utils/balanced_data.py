import pandas as pd
import sys

'''
Script destinado a balancear conjuntos de datos mediante undersampling.
Se genera un conjunto de datos donde se mantienen todos los registros
de la clase minoritaria y se obtiene una muestra aleatoria del mismo número
de registros de la otra clase.
Obtiene la ruta de origen de los datos y el destino del nuevo conjunto
generado como parámetros en línea de comandos. Admite únicamente ficheros CSV.
Ejemplo de uso:
py balanced_data.py "..\data\my_dataset.csv" "..\data\my_new_dataset.csv"
'''

# Recoger parámetros en línea de comandos
# Esperado: <ruta origen dataset> <ruta destino dataset>
args = sys.argv

full_data = pd.read_csv(args[1], low_memory=False)
# Filtramos los registros donde label es 0
df_normal = full_data[full_data['label'] == 0]

# Obtenemos el número de registros con label a 0
num_normal = len(df_normal)

# Filtramos los registros donde label no es 0
df_not_normal = full_data[full_data['label'] != 0]

# Obtenemos una muestra aleatoria de registros con label diferente de 0 del mismo tamaño
df_not_normal_sample = df_not_normal.sample(num_normal)

# Concatenamos los dos DataFrames para obtener el nuevo dataset
full_data = pd.concat([df_normal, df_not_normal_sample])

full_data.to_csv(args[2], index=False)