import pandas as pd
import sys
from data_util import split_data
# Ejemplo de uso: python split_data.py "..\data\data.csv" 0.8 label "\data\train_data.csv" ["..\data\test_data.csv"]
# Recoger argumentos en línea de comandos
# Se espera el fichero a dividir, porcentaje de entrenamiento, columna de objetivos y los dos ficheros, train y test
args = sys.argv

# Cargar dataset
data = pd.read_csv(args[1], low_memory=False)
# Particionar el dataset
train_data, test_data = split_data(data,float(args[2]),data[args[3]])
# Guardar los datasets
# Si se pasan 5 argumentos se guarda solo la partición declarada como de entrenamiento
if len(args) == 6:     
    pd.DataFrame(test_data).to_csv(args[5],index=False)
pd.DataFrame(train_data).to_csv(args[4],index=False)

