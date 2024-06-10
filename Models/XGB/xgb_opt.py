import pickle
from sklearn.model_selection import train_test_split, RandomizedSearchCV
import xgboost as xgb
import pandas as pd
import os, sys, time
import numpy as np

# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

# Carga de datos
dtrain = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+'bal_train_norm.csv')
dtest = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+'bal_test_norm.csv')

# Obtener una pequeña partición de evaluación a partir del conjunto de test
#dtest, deval = train_test_split(dtest,train_size=0.8,stratify=dtest['label'])

train_targets = dtrain.pop('label')
#eval_targets = deval.pop('label')
test_targets = dtest.pop('label')

# Convertir a DMatrix
#dtrain = xgb.DMatrix(dtrain, label=train_targets)
#deval = xgb.DMatrix(deval, label=eval_targets)
#dtest = xgb.DMatrix(dtest, label=test_targets)

# Malla de parámetros 

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



model = xgb.XGBClassifier()
print('Creando RandomizedSearchCV para optimización de hiperparámetros...')
rs_model = RandomizedSearchCV(model, params, n_iter=15, cv=2,n_jobs=4)
                           

#Obtener mejores atributos
print('Obteniendo la mejor combinación de atributos')
t_ini = time.time()
rs_model.fit(dtrain,train_targets)  
t_fin = time.time() - t_ini
best_xgb = rs_model.best_params_
print(f'Mejores parámetros: {best_xgb}')
print(f'Tiempo de obtención de parámetros optimizados: {t_fin}')
print()
print('Evaluando el modelo')
t_ini = time.time()
predictions = rs_model.predict(dtest)
t_fin = time.time() - t_ini
print('Obteniendo resultados:')
#Obtener métricas
metricas = model_utils.metrics(test_targets,predictions)
# Imprimir tiempo de entrenamiento y métricas
print(f'Tiempo de entrenamiento: {t_fin}')
print('Resultados:')

[print(i) for i in metricas]

model_utils.pr_roc_curves(predictions,test_targets)

'''
# Guardar el modelo entrenado
with open('xgb_model_attSel_opt.pkl', 'wb') as file:
    pickle.dump(rs_model, file)
'''