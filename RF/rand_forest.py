
import os, pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report

# Carga de datos
train_data = pd.read_csv('..' + os.sep + 'data' + os.sep + 'train_network.csv')
test_data = pd.read_csv('..' + os.sep + 'data' + os.sep + 'test_network.csv')
y_train = train_data.pop('label')
y_test = test_data.pop('label')

# Obtener pesos de las clases
class_0 = sum(y_train==0)
class_1 = sum(y_train==1)

class_w = {0: 1/(2*class_0), 1: 1/(2*class_1)}

# Eliminar algunas caracteristicas más. Esto se realizará en posteriormente en el preprocesado de los datasets
cols_to_drop = [col for col in train_data.columns if col.startswith('dst_ip_') or col.startswith('src_ip_') or col in ['src_port', 'dst_port', 'ts']]
# Eliminar las columnas
train_data = train_data.drop(columns=cols_to_drop)
test_data = test_data.drop(columns=cols_to_drop)
# Crear el modelo
rfc = RandomForestClassifier(class_weight=class_w, max_depth=10)
# Entrenar el modelo
rfc.fit(train_data, y_train)

# Predecir con el conjunto de test
predictions = rfc.predict(test_data)
predictions = predictions.astype(int)

# Obtener matriz de confusión
cm = confusion_matrix(y_test, predictions)
disp = ConfusionMatrixDisplay(confusion_matrix=cm)
disp.plot()
plt.show()

# Obtener el reporte de test
report = classification_report(y_test, predictions)
print("Reporte de clasificación:")
print(report)

'''
# Guardar el modelo entrenado
with open('rfc_model.pkl', 'wb') as file:
    pickle.dump(rfc, file)
'''