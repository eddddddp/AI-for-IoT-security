import pandas as pd
import sys

# Recoger parámetros en línea de comandos
# Esperado: <ruta dataset desbalanceado> <ruta dataset generado>
args = sys.argv

full_data = pd.read_csv(args[1], low_memory=False)
# Filtramos los registros donde 'type' es 'normal'
df_normal = full_data[full_data['label'] == 0]

# Obtenemos el número de registros 'normales'
num_normal = len(df_normal)

# Filtramos los registros donde 'type' no es 'normal'
df_not_normal = full_data[full_data['label'] != 0]

# Obtenemos una muestra aleatoria de registros no 'normales' del mismo tamaño que los 'normales'
df_not_normal_sample = df_not_normal.sample(num_normal)

# Concatenamos los dos DataFrames para obtener el nuevo dataset
full_data = pd.concat([df_normal, df_not_normal_sample])

full_data.to_csv(args[2], index=False)