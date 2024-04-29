from sklearn.model_selection import train_test_split
import xgboost as xgb
import pandas as pd
import os, sys, time

# Agregar directorio padre a al path de búsqueda
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

# Carga de datos
dtrain = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+'balanced_train.csv')

# Obtener partición de evaluación a partir del conjunto de entrenamiento
dtrain, deval = train_test_split(dtrain,train_size=0.8,stratify=dtrain['label'])
dtest = pd.read_csv('..'+os.sep+'..'+os.sep+'data'+os.sep+'balanced_test.csv')
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
    'colsample_bytree': 0.5, # Porcentage de características usadas para construir cada árbol
    'learning_rate': 0.1, # Tasa de aprendizaje
    'max_depth': 100, 
    'verbosity': 2, # 0 (silent), 1 (warning), 2 (info), 3 (debug)
    'gamma': 0, # Parámetro gamma. Controla la reducción de perdida mínima requerida para realizar una nueva división en un nodo del árbol
    'alpha': 0, # Término de regularización L1
    'lambda': 0, # Término de regularización L2 
    'device' : 'cuda', # Entrenamiento en GPU CUDA    
}

evaluation = [(deval, "eval"), (dtrain, "train")]
# Obtener tiempo de inicio
t_ini = time.time()
# Entrenar modelo
# num_boost_round indica el número de rondas de boosting
# early_stopping_rounds indica el número de rondas sin reducción del objetivo para realizar una parada anticipada
xg_reg = xgb.train(params=params, dtrain=dtrain, num_boost_round=500, early_stopping_rounds=5, evals=evaluation)
# Obtener tiempo de entrenamiento
t_train = time.time()-t_ini
print(f'Tiempo de entrenamiento: {t_train:2f}')
#Evaluar modelo con datos de test
predictions = xg_reg.predict(dtest)
# Para valores de probabilidad de clase 1 mayor que 0.5 se predice la calse 1 y 0 en cualquier otro caso
predictions = [1 if i > .5 else 0 for i in predictions]
# Obtener metricas de evaluación
metricas = model_utils.metrics(test_targets, predictions)
[print(i) for i in metricas]
'''
# Guardar el modelo
xg_reg.save_model("xgboost.model")
'''