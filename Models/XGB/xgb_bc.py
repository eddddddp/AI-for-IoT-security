from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import os, sys, time
import pickle
# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
Script que instancia, entrena y valida un booster de árboles de decisión equilibrados.
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

# Obtener una pequeña partición de evaluación a partir del conjunto de test
dtest, deval = train_test_split(dtest,train_size=0.9,stratify=dtest['label'])

# Obtener etiquetas o targets
train_targets = dtrain.pop('label')
eval_targets = deval.pop('label')
test_targets = dtest.pop('label')

# Convertir a DMatrix
dtrain = xgb.DMatrix(dtrain, label=train_targets)
deval = xgb.DMatrix(deval, label=eval_targets)
dtest = xgb.DMatrix(dtest, label=test_targets)

# Parámetros para XGBoost
params = {
    'booster':'gbtree', # Booster General Balanced Trees
    'objective':'binary:logistic', # Clasificación binaria
    'colsample_bytree': 0.6, # Porcentage de características usadas para construir cada árbol
    'learning_rate': 0.025, # Tasa de aprendizaje
    'max_depth': 20, # Profundidad máxima de árbol
    'verbosity': 1, # 0 (silent), 1 (warning), 2 (info), 3 (debug)        
    'device' : 'cuda', # Entrenamiento en GPU CUDA  
    'eval_metric':'error'  
}

# Crear el modelo
print('Creando el modelo...')
model = xgb.XGBClassifier()
evaluation = [(deval, "eval"), (dtrain, "train")]
# Obtener tiempo de inicio
t_ini = time.time()

# Entrenar modelo
print('Entrenando el modelo...')
xgb_model = xgb.train(params=params, dtrain=dtrain, num_boost_round=300, evals=evaluation)
# Obtener tiempo de entrenamiento
t_train = time.time()-t_ini
print(f'Tiempo de entrenamiento: {t_train:2f}')

#Evaluar modelo con datos de test
print('Evaluando el modelo...')
predictions = xgb_model.predict(dtest)
# Para valores de probabilidad de clase 1 mayor que 0.5 se predice la clase 1
predictions = [1 if i > .5 else 0 for i in predictions]

# Obtener y mostrar métricas de rendimiento
metricas = model_utils.metrics(test_targets, predictions)
print('Resultados:')
[print(i) for i in metricas]

# Mostrar curvas PR y ROC
model_utils.pr_roc_curves(predictions, test_targets)


# Guardar el modelo. Descomentar para guardar el modelo.
'''
model_utils.save_model('..' + os.sep + 'Modelos entrenados' + os.sep + 
                       'xgb_model_bc_error.pkl', xgb_model)
'''
