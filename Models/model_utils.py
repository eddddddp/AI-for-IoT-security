from sklearn.metrics import confusion_matrix, accuracy_score, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


def metrics(objetivo, prediction):
    cm = confusion_matrix(objetivo, prediction)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.show()
    score = accuracy_score(objetivo, prediction)
    reporte = classification_report(objetivo, prediction, digits=6)
    metricas = [cm, score, reporte]
    return(metricas)