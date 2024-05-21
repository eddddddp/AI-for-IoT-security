import pandas as pd
from data_util import one_hot_encode
import sys

from sklearn.preprocessing import LabelEncoder
# Recoger argumentos en línea de comandos
# Esperado: <input_filename> <data_output_filename> <flag si usa type o label>. Si flag true se usa type en cualquier otro caso label
#  Ejemplo: python prepare_ipv4_data.py "..\data\10_per_data.csv" "..\data\10_processed.csv" true/false
args = sys.argv

if len(args) != 4:
    print(f'Se esperaban 3 argumentos pero se obtuvieron {len(args)-1}')
    print('Se esperaba <src data> <dst data> <target_flag>')
    exit(-1)

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)
# Eliminar filas en las que src_bytes vale 0.0.0.0
data = data[data['src_bytes'] != '0.0.0.0']

# Obtener características y separar numéricas y texto o str
data_str = data[['proto','service','conn_state']]
# Convertir a float las numéricas
data_num = data[['dst_bytes','dst_ip_bytes','missed_bytes','dst_pkts','duration','src_bytes','src_ip_bytes','src_pkts']].astype(float)

#data_str = data_str.drop('uid',axis=1)
if args[3] == 'true':
    # Crear el codificador
    le = LabelEncoder()
    targets = data[['type']]
    # Ajustar y transformar los datos
    targets['type_cod'] = le.fit_transform(targets['type']).astype(float)

    # Obtener correspondencia entre las clases y los valores asignados
    print(dict(zip(le.classes_, le.transform(le.classes_))))
    
else:   
    targets = data[['label']].astype(float)
# Codificar data_str
data_encoded = one_hot_encode(data_str)
# Reiniciar los índices de las filas
data_num = data_num.reset_index(drop=True)
data_encoded = data_encoded.reset_index(drop=True)

'''
# Normalizar data_num
data_num = pd.DataFrame(normalize(data_num.values),columns=data_num.columns)
'''

# Reconstruir el dataset
full_data = pd.concat([targets,data_num,data_encoded], axis=1)

# Eliminar filas con NaN
full_data.dropna(subset=full_data.columns, inplace=True)

# Guardar csv procesado
full_data.to_csv(args[2],index=False)