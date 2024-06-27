import os, pickle
from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import auc, confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline

'''
Función que dado un conjunto de etiquetas y de predicciones devuelve
las métricas del reporte de clasificación y la matriz de confusión.
Parameters: Objetivos, predicciones.
Return: Lista de métricas de rendimiento.
'''
def metrics(objetivo, prediction):
    # Obtener matriz de confusión
    cm = confusion_matrix(objetivo, prediction)
    # Mostrar matriz de confusión
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    # Obtener accuracy
    score = accuracy_score(objetivo, prediction)
    # Obtener reporte de clasificación
    reporte = classification_report(objetivo, prediction, digits=6)
    # Añadir todas las métricas a la lista
    metricas = [cm, score, reporte]
    # Retornar lista de métricas
    return(metricas)

'''
Función que dado un modelo de clasificación devuelve un pipeline
para selección de los k mejores atributos. Se tiene un valor de
k por defecto de 10 que se toma en caso de no indicar este parámetro.
Parameters: Modelo, núm. atributos.
Return: Pipeline.
'''
def make_pipeline(model, k=10):
    return Pipeline([
        ('selector', SelectKBest(chi2,k=k)),
        ('clasificador', model)
    ])

'''
Función que dado un conjunto de predicciones y etiquetas
muestra las curvas PR y ROC para el modelo del que 
se obtienen las predicciones.
'''
def pr_roc_curves(predictions, y_test):

    # Obtener curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, predictions)

    # Crear y mostrar curva PR
    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Curva PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='best')
    plt.show()

    # Obtener curva ROC y AUC (area under curve)
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

    # Crear y mostrar curva ROC y AUC
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label='Curva ROC (area = %0.2f)' % roc_auc)
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Tasa de Falsos Positivos')
    plt.ylabel('Tasa de Verdaderos Positivos')
    plt.title('Curva ROC')
    plt.legend(loc="lower right")
    plt.show()

'''
Función para guardar modelos como objetos serializados
'''
def save_model(path_file, model):
    with open(path_file, 'wb') as file:
        pickle.dump(model, file)

'''
Función para la carga de modelos guardados como objetos serializados.
'''
def load_model(path_file):
    with open(path_file, 'rb') as file:
        model = pickle.load(file)
    return model