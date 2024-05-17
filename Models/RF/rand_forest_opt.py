import os, sys, time, pickle
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report, f1_score, make_scorer, precision_score, recall_score, accuracy_score
from sklearn.model_selection import RandomizedSearchCV, cross_validate
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
En este módulo se implementa un modelo random forest. 
Se carga el conjunto de datos preprocesado, se crea
un pipeline para la selección de atributos con selectKBest,
y se optimizan los hiperparámetros del modelo y el número de 
características óptimo a seleccionar. Se evalua el modelo
obteniendo las métricas obtenidas en las predicciones con el
 conjunto de test. Finalmente se entrena y se guarda el modelo entrenado.
'''
# Carga de datos
print('Cargando datos...')
train_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'balanced_train.csv')
test_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 'balanced_test.csv')
y_train = train_data.pop('label')
y_test = test_data.pop('label')

print('Obteniendo pesos de las clases...')
# Obtener pesos de las clases
class_0 = sum(y_train==0)
class_1 = sum(y_train==1)

class_w = {0: 1/(2*class_0), 1: 1/(2*class_1)}

# Crear diccionario de hiperparámetros
params = dict(selector__k=[10, 15, 20, 25, 30, 35, 40, 47],
                      clasificador__n_estimators = np.linspace(200,300,15).astype(int),
                      clasificador__max_depth = np.linspace(30,70,11).astype(int),
                      clasificador__criterion = ['gini', 'entropy', 'log_loss'],
                      clasificador__bootstrap = [True, False])

print('Creando el modelo y el pipeline...')
# Crear el modelo
rfc = RandomForestClassifier(class_weight=class_w, random_state=24, n_jobs = 12)

# Crear pipeline para selección de atributos
pipeline_rfc = model_utils.make_pipeline(rfc)

# Crear gridSearch para optimización de hiperparámetros
print('Creando gridSearch para optimización de hiperparámetros...')
grid_rfc_pipe = RandomizedSearchCV(pipeline_rfc, params, cv=5, n_iter=5)
#Obtener mejores atributos
print('Obteniendo la mejor combinación de atributos')
t_ini_rand = time.time()
grid_rfc_pipe.fit(train_data,y_train)
best_rfc = grid_rfc_pipe.best_params_
t_fin_rand = time.time() -t_ini_rand
print(f'Mejores parámetros: {best_rfc}')
print(f'Tiempo de obtención de parámetros optimizados: {t_fin_rand}')

# Entrenar y validar el modelo con los mejores hiperparámetros
# Crear modelo con hiperparámetros optimizados
print('Creando modelo optimizado y pipeline...')
rfc_opt = RandomForestClassifier(n_estimators=best_rfc['clasificador__n_estimators'],max_depth=best_rfc['clasificador__max_depth'], criterion=best_rfc['clasificador__criterion'], bootstrap=best_rfc['clasificador__bootstrap'] ,class_weight=class_w, random_state=24)

# Crear pipeline para selección de atributos con el modelo optimizado usando el valor de k obtenido en la optimización
pipe_rfc_opt = model_utils.make_pipeline(rfc_opt,best_rfc['selector__k'])

print('Entrenando pipeline...')
# Obtener tiempo de inicio
t_ini = time.time()
# Entrenar el pipeline
pipe_rfc_opt.fit(train_data,y_train)
# Obtener tiempo de entrenamiento
t_train = time.time() - t_ini
# Validar el pipeline con datos de test
y_pred = pipe_rfc_opt.predict(test_data)
print('Entrenado, obteniendo métricas...')
#Obtener métricas
metricas = model_utils.metrics(y_test,y_pred)
# Imprimir tiempo de entrenamiento y métricas
print(f'Tiempo de entrenamiento: {t_train}')
print('Resultados:')
[print(i) for i in metricas]

'''
# Guardar el modelo entrenado
with open('rfc_model_attSel_opt.pkl', 'wb') as file:
    pickle.dump(pipe_rfc_opt, file)
'''