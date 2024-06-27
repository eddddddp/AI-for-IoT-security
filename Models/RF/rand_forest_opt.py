import os, sys, time, pickle
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
Script que instancia, entrena y valida un modelo Random Forest, implementando selección de
atributos y optimización de hiper-parámetros mediante busqueda aleatoria.
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
estas líneas para evitar sobrescribit los modelos guardados al realizar pruebas que 
no se quieran guardar.
'''
# Carga de datos
print('Cargando datos...')
train_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 
                         'bal_train_norm.csv')
test_data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 
                        'bal_test_norm.csv')

# Obtener etiquetas o targets
y_train = train_data.pop('label')
y_test = test_data.pop('label')

# Obtener pesos de las clases
print('Obteniendo pesos de las clases...')
class_0 = sum(y_train==0)
class_1 = sum(y_train==1)

class_w = {0: 1/(2*class_0), 1: 1/(2*class_1)}

# Crear diccionario de hiperparámetros
params = dict(selector__k=[30, 35, 40, 44, 45, 46, 47],
                      clasificador__n_estimators = np.linspace(200,300,8).astype(int),
                      clasificador__max_depth = np.linspace(10,60,6).astype(int),
                      clasificador__criterion = ['gini', 'entropy', 'log_loss'])

print('Creando el modelo y el pipeline...')
# Crear el modelo
rfc = RandomForestClassifier(class_weight=class_w, random_state=24)

# Crear pipeline para selección de atributos
pipeline_rfc = model_utils.make_pipeline(rfc)

# Crear gridSearch para optimización de hiperparámetros
print('Creando RandomizedSearchCV para optimización de hiperparámetros...')
grid_rfc_pipe = RandomizedSearchCV(pipeline_rfc, params, cv=2, n_iter=15, n_jobs=10)
#Obtener mejores atributos
print('Obteniendo la mejor combinación de atributos...')
t_ini = time.time()
grid_rfc_pipe.fit(train_data,y_train)
t_fin = time.time() - t_ini
best_rfc = grid_rfc_pipe.best_params_
print(f'Mejores parámetros: {best_rfc}')
print(f'Tiempo de obtención de parámetros optimizados: {t_fin}')
print()
print('Evaluando el modelo...')
predictions = grid_rfc_pipe.predict(test_data)

# Obtener y mostrar métricas
metricas = model_utils.metrics(y_test,predictions)
print('Resultados:')
[print(i) for i in metricas]

# Mostrar curvas PR y ROC
model_utils.pr_roc_curves(predictions,y_test)

# Guardar el modelo entrenado. Descomentar para guardar el modelo.
'''
model_utils.save_model('..' + os.sep + 'Modelos entrenados' + os.sep + 
                       'rfc_model_attSel_opt.pkl', grid_rfc_pipe)
'''