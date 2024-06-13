import os, sys, time, pickle
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
Script que instancia, entrena y valida un modelo Random Forest.
Tras el entrenamiento y la validación se muestran las siguientes métricas de rendimiento:
- Accuracy
- Precision
- Recall
- F1-score
- Matriz de confusión
- Macro average
- Weighted average
- Curva PR
- Curva ROC
Para guardar el modelo tras el entrenamiento, descomentar las últimas líneas del código
antes de la ejecución del script. Tras la ejecución se recomienda volver a comentar
estas líneas para evitar sobreescribir los modelos guardados al realizar pruebas que 
no se quieran guardar.
'''

# Carga de datos
print('Cargando datos...')
train_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'bal_train_norm.csv')
test_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'bal_test_norm.csv')

# Obtener etiquetas targets
y_train = train_data.pop('label')
y_test = test_data.pop('label')

# Obtener pesos de las clases
print('Obteniendo pesos de las clases...')
class_0 = sum(y_train==0)
class_1 = sum(y_train==1)

class_w = {0: 1/(2*class_0), 1: 1/(2*class_1)}

# Crear el modelo
print('Creando el modelo...')
rfc = RandomForestClassifier(n_estimators=250, class_weight=class_w, max_depth=30, random_state=24, criterion='log_loss', n_jobs=8)

#Obtener tiempo inicial
t_ini = time.time()
# Entrenar el modelo
print('Entrenando el modelo...')
rfc.fit(train_data, y_train)
# Obtener tiempo de entrenamiento
t_train = time.time() - t_ini

# Predecir con el conjunto de test
print('evaluando el modelo...')
predictions = rfc.predict(test_data)
predictions = predictions.astype(int)

#Imprimir tiempo de entrenamiento
print()
print(f'Tiempo de entrenamiento: {t_train:.2f} segundos')

# Mostrar métricas de rendimiento
metricas = model_utils.metrics(y_test, predictions)
print('Resultados:')
[print(i) for i in metricas]

# Mostrar curvas PR y ROC
model_utils.pr_roc_curves(predictions, y_test)

# Guardar el modelo entrenado. Descomentar para guardar el modelo.
'''
with open('rfc_model_bc_entropy.pkl', 'wb') as file:
    pickle.dump(rfc, file)
'''