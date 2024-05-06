import os, sys, time, pickle
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

# Carga de datos
train_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'balanced_train.csv')
test_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'balanced_test.csv')
y_train = train_data.pop('label')
y_test = test_data.pop('label')

# Obtener pesos de las clases
class_0 = sum(y_train==0)
class_1 = sum(y_train==1)

class_w = {0: 1/(2*class_0), 1: 1/(2*class_1)}

# Crear el modelo
rfc = RandomForestClassifier(n_estimators=241, class_weight=class_w, max_depth=49,  random_state=24)
#Obtener tiempo incial
t_ini = time.time()
# Entrenar el modelo
rfc.fit(train_data, y_train)
# Obtener tiempo de entrenamiento
t_train = time.time() - t_ini
# Predecir con el conjunto de test
predictions = rfc.predict(test_data)
predictions = predictions.astype(int)
'''
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
#Imprimir tiempo de entrenamiento
print()
print(f'Tiempo de entrenamiento: {t_train:.2f} segundos')
metricas = model_utils.metrics(y_test, predictions)
[print(i) for i in metricas]
'''
# Guardar el modelo entrenado
with open('rfc_model.pkl', 'wb') as file:
    pickle.dump(rfc, file)
'''