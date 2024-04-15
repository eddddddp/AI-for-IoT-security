import pandas as pd
from data_util import ipv4_to_octets, one_hot_encode
import sys
from keras.utils import normalize
# Recoger argumentos en línea de comandos
# Esperado: <input_filename> <data_output_filename>
# python prepare_ipv4_data.py "..\data\10_per_data.csv" "..\data\10_processed.csv"
args = sys.argv

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)
# Filtrar las filas donde 'src_ip' y 'dst_ip' contienen un punto para eliminar entradas no válidas como '0'
# y las IPv6
data = data[data['src_ip'].str.contains('\.') & data['dst_ip'].str.contains('\.')]
# Obtener características y separar numéricas y texto o str
data_str = data[['src_ip','dst_ip','proto','service','conn_state']]
# Convertir a float las numéricas
data_num = data[['src_port','dst_port','dst_bytes','dst_ip_bytes','missed_bytes',
                 'dst_pkts','duration','src_bytes','src_ip_bytes','src_pkts','label']].astype(float)
# Crear características para cada octeto de las IP
for col in ['dst_ip', 'src_ip']:    
    # Crear un DataFrame con las columnas de octetos
    octets = data[col].apply(ipv4_to_octets).apply(pd.Series)
    # Cambiar el tipo de datos a float
    octets = octets.astype(float)
    # Renombrar las características
    octets.columns = [f'{col}_{i}' for i in range(4)]
    # Unir el DataFrame de octetos y de caracteristicas numéricas
    data_num = pd.concat([data_num, octets], axis=1)
# Eliminar las columnas src_ip y dst_ip
data_str = data_str.drop(['src_ip','dst_ip'], axis=1)
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
# Guardar el dataset en formato csv
full_data.to_csv(args[2],index=False)