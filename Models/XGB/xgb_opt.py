import pickle
from sklearn.model_selection import RandomizedSearchCV
import xgboost as xgb
import pandas as pd
import os, sys, time
import numpy as np
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
Script que instancia, entrena y valida un booster de árboles de decisión equilibrados,
implementando la selección de atributos y ooptimización de hiper-parámetros mediante
búsqueda aleatoria sobre una malla definida de hiper-parámetros.
Tras el entrenamiento y validación se muestran las siguientes métricas de rendimiento:
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
dtrain = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+
                     'bal_train_norm.csv')
dtest = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+
                    'bal_test_norm.csv')

# Obtener etiquetas o targets
train_targets = dtrain.pop('label')
test_targets = dtest.pop('label')

# Malla de hiper-parámetros 
params = {
    'booster':['gbtree'], # Booster General Balanced Trees
    'objective':['binary:logistic'], # Clasificación binaria
    'colsample_bytree': np.linspace(0.3,0.7,5), # Porcentage de características usadas para construir cada árbol
    'learning_rate': np.linspace(0.01,0.5,5), # Tasa de aprendizaje
    'max_depth': np.linspace(10,60,6).astype(int), # Profundidad máxima de árbol
    'verbosity': [0], # 0 (silent), 1 (warning), 2 (info), 3 (debug)        
    'device' : ['cuda'], # Entrenamiento en GPU CUDA
    'num_boost_round':np.linspace(200,300,8).astype(int),
    'eval_metric': ['log_loss', 'error']   
}

# Crear el modelo
print('Creando el modelo...')
model = xgb.XGBClassifier()

# Crear RandomizedSearchCV
print('Creando RandomizedSearchCV para optimización de hiperparámetros...')
rs_model = RandomizedSearchCV(model, params, n_iter=15, cv=2,n_jobs=4)                           

# Obtener mejores atributos
print('Obteniendo la mejor combinación de parámetros...')
t_ini = time.time()
rs_model.fit(dtrain,train_targets)  
t_fin = time.time() - t_ini
best_xgb = rs_model.best_params_
print(f'Mejores parámetros: {best_xgb}')
print(f'Tiempo de obtención de parámetros optimizados: {t_fin}')
print()

print('Evaluando el modelo...')
predictions = rs_model.predict(dtest)

# Obtener y mostrar métricas
metricas = model_utils.metrics(test_targets,predictions)
print('Resultados:')
[print(i) for i in metricas]

# Mostrar curvas PR y ROC
model_utils.pr_roc_curves(predictions,test_targets)


# Guardar el modelo. Descomentar para guardar el modelo.
'''
model_utils.save_model('..' + os.sep + 'Modelos entrenados' + os.sep + 
                       'xgb_model_attSel_opt.pkl', rs_model)
'''