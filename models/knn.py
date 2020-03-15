import numpy as np

from dataset_preparation.preprocess_data import Dataset
from models.evaluate_model import evaluate_model
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier 


def knn(dataset, features):
    X, Y = dataset(features)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2.0/10, random_state=8)

    model = KNeighborsClassifier(algorithm="auto", n_neighbors=20)
    evaluate_model(model, X_train, X_test, Y_train, Y_test)


if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    knn(dataset, ["melspectogram", "rms", "spectral_bandwidth", "spectral_centroid", "spectral_rolloff"])
    