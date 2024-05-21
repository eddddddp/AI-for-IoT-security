import pandas as pd
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import sys

def normalizeData(traindata, testdata):
    scaler = MinMaxScaler()   
    train_scaled = scaler.fit_transform(traindata)
    test_scaled = scaler.transform(testdata)
    train_scaled = pd.DataFrame(train_scaled, columns=traindata.columns)
    test_scaled = pd.DataFrame(train_scaled, columns=testdata.columns)
    return train_scaled, test_scaled

def standarizeData(traindata, testdata):
    scaler = StandardScaler()   
    train_scaled = scaler.fit_transform(traindata)
    test_scaled = scaler.transform(testdata)
    train_scaled = pd.DataFrame(train_scaled, columns=traindata.columns)
    test_scaled = pd.DataFrame(train_scaled, columns=testdata.columns)
    return train_scaled, test_scaled

# Obtener argumentos
args = sys.argv
if len(args) != 6:
    print(f'Número de argumentos incorrecto, se esperaban 5 pero se obtuvieron {len(args)-1}.')
    print('Se espera <src train_data> <src test_data> <dst train_norm_data> <dst test_norm_data> <normalize/standarize>')
    exit(-1)

# Cargar conjuntos de datos
train_data = pd.read_csv(args[1])
test_data = pd.read_csv(args[2])

if args[5] == 'normalize':
    train_data, test_data = normalizeData(train_data, test_data)
elif args[5] == 'standarize':
        train_data, test_data = standarizeData(train_data, test_data)
else:
     print(f'Parámetro de normalización incorrecto, se esperaba "normalize" o "standarize", pero se obtuvo {args[5]}')
     exit(-2)

train_data.to_csv(args[3])
test_data.to_csv(args[4])