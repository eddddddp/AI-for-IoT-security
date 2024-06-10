from sklearn.feature_selection import SelectKBest, chi2
from sklearn.metrics import auc, confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay, precision_recall_curve, roc_curve
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline


def metrics(objetivo, prediction):
    cm = confusion_matrix(objetivo, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    score = accuracy_score(objetivo, prediction)
    reporte = classification_report(objetivo, prediction, digits=6)
    metricas = [cm, score, reporte]
    return(metricas)

def make_pipeline(model, k=10):
    return Pipeline([
        ('selector', SelectKBest(chi2,k=k)),
        ('clasificador', model)
    ])

def pr_roc_curves(predictions, y_test):
    # Curva Precision-Recall
    precision, recall, _ = precision_recall_curve(y_test, predictions)

    plt.figure(figsize=(8, 6))
    plt.plot(recall, precision, label='Curva PR')
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Curva Precision-Recall')
    plt.legend(loc='best')
    plt.show()

    # Curva ROC
    fpr, tpr, _ = roc_curve(y_test, predictions)
    roc_auc = auc(fpr, tpr)

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
