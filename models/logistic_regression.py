import numpy as np
import matplotlib.pyplot as plt

from dataset_preparation.preprocess_data import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.metrics import plot_confusion_matrix

if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    X, Y = dataset(["rms"])
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/7)

    model = LogisticRegression(solver="saga")
    classifier = model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    for i in range(len(Y_test)):
        if Y_test[i] != Y_pred[i] and Y_test[i] == "sax":
            print(Y_test[i] + " " + Y_pred[i])
    plot_confusion_matrix(
        classifier,
        X_test,
        Y_test,
        cmap=plt.cm.Blues,
        normalize=None)

    plt.show()
    # count_misclassified = (Y_test != Y_pred).sum()
    # print("Misclassified samples: {}".format(count_misclassified))
    # accuracy = metrics.accuracy_score(Y_test, Y_pred)
    # print("Accuracy: {:.2f}".format(accuracy))
    
    
    
    print("Nice")
                
