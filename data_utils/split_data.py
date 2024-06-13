import pandas as pd
import sys
from data_util import split_data
'''
Script para dividir el conjunto de datos en particiones de entrenamiento y validación o test.
Recibe en línea de comandos la ruta de los datos de origen, la fracción en tanto por uno
para entrenamiento, la característica de estratificación, el destino de la partición de 
entrenamiento y, opcionalmente, el destino de la particion de validación o test. Este script
también puede ser usado para obtener un subconjunto aleatorio de un conjunto de datos, manteniendo
la proporción entre las clases.
Ejemplos de uso:
py split_data.py "..\data\my_data.csv" 0.8 label "..\data\my_train_data.csv" "..\data\my_test_data"
py split_data.py "..\data\my_data.csv" 0.5 label "..\data\my_half_data.csv" 
'''

# Recoger argumentos en línea de comandos
args = sys.argv

# Comprobar el número de argumentos
if len(args) != 5 and len(args) != 6:
    print(f'Número de argumentos incorrecto, se esperaban 4 o 5 pero se obtuvieron {len(args)-1}')
    print('Se espera <src data> <train_size> <stratify col> <dst train_data> [<dst test_data>]')
    exit(-1)

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)

# Crear particiones
train_data, test_data = split_data(data,float(args[2]),data[args[3]])

# Guardar los datasets
# Si se pasan 5 argumentos se guarda solo la partición declarada como de entrenamiento
if len(args) == 6:     
    pd.DataFrame(test_data).to_csv(args[5],index=False)
pd.DataFrame(train_data).to_csv(args[4],index=False)

