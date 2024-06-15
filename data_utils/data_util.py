from sklearn.preprocessing import MinMaxScaler, OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd

'''
Función que dado un conjunto de datos que solo contiene atributos categóricos
los codifica como valores numéricos mediante la técnica de one-hot encoding.
Parameters: Dataset con características a codificar.
Return: Dataset con características codificadas.
'''
def one_hot_encode(data):
    # Obtener encoder
    encoder = OneHotEncoder(sparse_output=False)
    # Codificar las columnas de data
    data_encoded = encoder.fit_transform(data)
    # Obtener los nombres de las nuevas columnas
    col_names = encoder.get_feature_names_out(data.columns)
    # Convertir a dataframe de pandas 
    data_encoded = pd.DataFrame(data_encoded, columns=col_names)
    # Convertir los datos codificados de booleanos a números reales
    data_encoded = data_encoded.astype(float)
    # Devolver el dataframe
    return data_encoded

'''
Función que dado un conjunto de datos, una fracción en tanto por uno de los datos
a usar en entrenamiento, y una columna de estatificación, devuelve las particiones de 
entrenamiento y validación en las proporciones indicadas de forma estratificada por 
la característica seleccionada.
Parameters: Dataset, fracción de particion de entrenamiento, característica de estratificación.
Return: Partición de entrenamiento y partición de test o validación.
'''
def split_data(data,train_percent,stratify_col):
    # Obtener particiones de entrenamiento y validación o test
    train_data, test_data = train_test_split(data,train_size=train_percent,stratify=stratify_col)
    return train_data, test_data

'''
Función que dadas las particiones de entrenamiento y test, realiza la normalización
de los valores de ambas particiones.
Parameters: Partición de entrenamiento, partición de validación.
Return: Partición de entrenamiento normalizada, partición de validación normalizada.
'''
def normalizeData(traindata, testdata):
    # Instanciar scaler
    scaler = MinMaxScaler()   
    # Normalizar partición de entrenamiento
    train_scaled = scaler.fit_transform(traindata)
    # Normalizar partición de validación
    test_scaled = scaler.transform(testdata)
    # Convertir particiones a DataFrame de pandas
    train_scaled = pd.DataFrame(train_scaled, columns=traindata.columns)
    test_scaled = pd.DataFrame(train_scaled, columns=testdata.columns)
    return train_scaled, test_scaled