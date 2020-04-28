from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

import matplotlib.pyplot as plt
import inspect


def evaluate_model(model, X_train, X_test, Y_train, Y_test):
    classifier = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)

    # plot_confusion_matrix(
        # classifier,
        # X_test,
        # Y_test,
        # cmap="Blues",
        # normalize=None)

    # plt.title(inspect.stack()[1][3])
    # plt.show()
    
    # count_misclassified = (Y_test != Y_pred).sum()
    # print("Misclassified samples: {}".format(count_misclassified))
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    # print("Accuracy: {:.2f}".format(accuracy))
    return accuracy