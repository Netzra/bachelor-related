import matplotlib.pyplot as plt
import numpy
from sklearn import metrics
#______________________________________________________________________________________________________________
# Notes
#
# reference https://www.w3schools.com/python/python_ml_confusion_matrix.asp
# https://en.wikipedia.org/wiki/Confusion_matrix
#
# TODO
# might implement Traits later
# might add distribution plotting if needed
#______________________________________________________________________________________________________________


class ConfusionMatrix:

    def __init__(self, actual, predicted):
        self.actual = actual
        self.predicted = predicted

    def plot_matrix(self):
        confusion_matrix = metrics.confusion_matrix(self.actual, self.predicted)
        cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=confusion_matrix, display_labels=[False, True])
        cm_display.plot()
        plt.show()

    def plot_distribution(self):
        pass

    def accuracy(self):
        print("Accuracy: " + str(metrics.accuracy_score(self.actual, self.predicted)))

    def precision(self):
        print("Precision: " + str(metrics.precision_score(self.actual, self.predicted)))

    def sensitivity(self):
        print("Sensitivity: " + str(metrics.recall_score(self.actual, self.predicted)))

    def specificity(self):
        print("Specificity: " + str(metrics.recall_score(self.actual, self.predicted, pos_label=0)))

    def f_score(self):
        print("f_score: " + str(metrics.f1_score(self.actual, self.predicted)))


#_____________________________________________________________________________________________________________________________
# For Debugging
#_____________________________________________________________________________________________________________________________


if __name__ == "__main__":
    test = ConfusionMatrix(
        actual=numpy.random.binomial(1, 0.9, size=1000),
        predicted=numpy.random.binomial(1, .9, size=1000)
    )
    test.plot_matrix()
    test.accuracy()
    test.precision()
    test.sensitivity()
    test.specificity()
    test.f_score()
