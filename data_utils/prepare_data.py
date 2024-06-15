import pandas as pd
import sys
from data_util import one_hot_encode

'''
Script para preparar el conjunto de datos. Recibe en línea de comandos la
ruta de origen de los datos y el destino donde se guardará el conjunto preparado.
Se seleccionan las características que se mantienen, se limpian los registros anómalos, 
se codifican mediante one-hot encoding las características categóricas y se eliminan
los registros que contienen nulos.
Ejemplo de uso:
py prepare_date.py "..\data\my_dataset.csv" "..\data\my_prepared_dataset.csv"
'''
# Recoger argumentos en línea de comandos
args = sys.argv

# Comprobar el número de parámetros recibidos
if len(args) != 3:
    print(f'Se esperaban 2 argumentos pero se obtuvieron {len(args)-1}')
    print('Se esperaba <src data> <dst data>')
    exit(-1)

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)

# Eliminar filas en las que src_bytes vale 0.0.0.0
data = data[data['src_bytes'] != '0.0.0.0']

# Obtener características y separar numéricas y texto o str
data_str = data[['proto','service','conn_state']]

# Convertir a float las numéricas
data_num = data[['dst_bytes','dst_ip_bytes','missed_bytes','dst_pkts','duration','src_bytes','src_ip_bytes','src_pkts']].astype(float)

# Obtener etiquetas o targets 
targets = data[['label']].astype(float)

# Codificar data_str
data_encoded = one_hot_encode(data_str)

# Reiniciar los índices de las filas
data_num = data_num.reset_index(drop=True)
data_encoded = data_encoded.reset_index(drop=True)

# Reconstruir el dataset
full_data = pd.concat([targets,data_num,data_encoded], axis=1)

# Eliminar filas con NaN
full_data.dropna(subset=full_data.columns, inplace=True)

# Guardar csv preparado
full_data.to_csv(args[2],index=False)