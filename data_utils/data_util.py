from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
import pandas as pd
import ipaddress

# Función para convertir la dirección IP a una lista de 16 octetos
def ipv4_to_octets(ip):
    try:
        ip = ipaddress.ip_address(ip)
        if ip.version == 4:
            # Devolver los octetos
            return list(ip.packed)        
    except ValueError:
        print('Se ha encontrado una IP no válida en el dataset')

# Función que realiza el one-hot encoding a las características del dataframe pasado por parámetro    
def one_hot_encode(data):
    # Obtener encoder. Se eliminan las columnas “dummy” con la opción drop
    encoder = OneHotEncoder(sparse_output=False, drop='first')
    # Codificar las columnas de data
    data_encoded = encoder.fit_transform(data)
    # Convertir a dataframe de pandas 
    data_encoded = pd.DataFrame(data_encoded)
    # Convertir los datos codificados de booleanos a números reales
    data_encoded = data_encoded.astype(float)
    # Devolver el dataframe
    return data_encoded

def split_data(data,train_percent,stratify_col):
    train_data, test_data = train_test_split(data,train_size=train_percent,stratify=stratify_col)
    return train_data, test_data