import pandas as pd
import sys
from data_util import normalizeData
'''
Script destinado a la normalización de los valores de las particiones
de entrenamiento y validación o test en formato CSV. Se pasan las rutas de origen y
destino como parámetros en línea de comandos.
Ejemplo de uso:
py normalize_data.py "..\data\my_train_set.csv" "..\data\my_test_set.csv" "..\data\my_train_set_normalized.csv" "..\data\my_test_set_normalized.csv"
'''
# Obtener argumentos
args = sys.argv
if len(args) != 5:
    print(f'Número de argumentos incorrecto, se esperaban 4 pero se obtuvieron {len(args)-1}.')
    print('Se espera <src train_data> <src test_data> <dst train_norm_data> <dst test_norm_data>')
    exit(-1)

# Cargar conjuntos de datos
train_data = pd.read_csv(args[1])
test_data = pd.read_csv(args[2])

# Normalizar valores de las particiones
train_data, test_data = normalizeData(train_data, test_data)

train_data.to_csv(args[3])
test_data.to_csv(args[4])