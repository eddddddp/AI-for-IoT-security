import pickle, os, sys, time
import pandas as pd
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import model_utils

'''
Módulo destinado a probar el funcionamiento de los modelos XGB preentrenados
que implementan optimización mediante búsqueda aleatoria.
Carga el modelo de la ruta y fichero indicado y se clasifica una pequeña 
muestra aleatoria de los registros usados en validación.
'''

# Cargar modelo
model = model_utils.load_model('..' + os.sep + 'Modelos entrenados' + os.sep +
                               'xgb_model_attSel_opt.pkl')

# Cargar datos
data = pd.read_csv('..' + os.sep + '..' + os.sep + 'data' + os.sep + 
                   'bal_test_norm.csv')

# Obtener muestra
sample_data = data.sample(1000)

# Extraer etiquetas
sample_targets = sample_data.pop('label')

# Clasificar la muestra
t_ini = time.time()
predictions = model.predict(sample_data)

# Para valores de probabilidad de clase 1 mayor que 0.5 se predice la clase 1
predictions = [1 if i > .5 else 0 for i in predictions]
t_fin = time.time() - t_ini

# Mostrar tiempo y resultados de la clasificación
print(f'Tiempo prediccion: {t_fin}')
metricas = model_utils.metrics(sample_targets, predictions)
[print(i) for i in metricas]