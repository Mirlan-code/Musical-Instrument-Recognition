import numpy as np
import matplotlib.pyplot as plt

from dataset_preparation.preprocess_data import Dataset
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics


if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    X, Y = dataset([ "rms"])
    print(X.shape)
    print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=1.0/7, random_state=122)

    model = LogisticRegression(solver="saga")
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    count_misclassified = (Y_test != Y_pred).sum()
    print("Misclassified samples: {}".format(count_misclassified))
    accuracy = metrics.accuracy_score(Y_test, Y_pred)
    print("Accuracy: {:.2f}".format(accuracy))
    
    
    
    print("Nice")
                
