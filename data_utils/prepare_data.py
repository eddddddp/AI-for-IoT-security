import pandas as pd
from data_util import one_hot_encode
import sys
from keras.utils import normalize
# Recoger argumentos en línea de comandos
# Esperado: <input_filename> <data_output_filename>
#  Ejemplo: python prepare_ipv4_data.py "..\data\10_per_data.csv" "..\data\10_processed.csv"
args = sys.argv

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)
# Eliminar filas en las que src_bytes vale 0.0.0.0
data = data[data['src_bytes'] != '0.0.0.0']
# Obtener características y separar numéricas y texto o str
data_str = data[['proto','service','conn_state']]
# Convertir a float las numéricas
data_num = data[['dst_bytes','dst_ip_bytes','missed_bytes','dst_pkts','duration','src_bytes','src_ip_bytes','src_pkts','label']].astype(float)
# Codificar data_str
data_encoded = one_hot_encode(data_str)
# Reiniciar los índices de las filas
data_num = data_num.reset_index(drop=True)
data_encoded = data_encoded.reset_index(drop=True)
# Normalizar data_num
targets = data_num.pop('label')
data_num = pd.DataFrame(normalize(data_num.values),columns=data_num.columns)
# Reconstruir el dataset
full_data = pd.concat([targets,data_num,data_encoded], axis=1)
# ELiminar filas en las que src_bytes vale 0.0.0.0 y convertir a float todas las características
data_num = data_num[data_num['src_bytes'] != '0.0.0.0'].astype(float)
# Guardar el dataset en formato csv
full_data.to_csv(args[2],index=False)