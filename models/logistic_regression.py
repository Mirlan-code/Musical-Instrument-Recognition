import numpy as np
import itertools 

from models.evaluate_model import evaluate_model
from dataset_preparation.preprocess_data import Dataset
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression


def logistic_regression(dataset, features):
    X, Y = dataset(features)
    print(X.shape)
    print(Y.shape)
    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=2.0/10)

    model = LogisticRegression(solver="saga", max_iter=300)
    return evaluate_model(model, X_train, X_test, Y_train, Y_test)

def findsubsets(s, n): 
    return [set(i) for i in itertools.combinations(s, n)]

if __name__ == "__main__":
    dataset = Dataset(path="dataset", reinitialize=False)
    s = [
        "chroma_stft",
        "chroma_cqt",
        "chroma_cens",
        "melspectogram",
        # "rms",
        # "spectral_centroid",
        "spectral_bandwidth",
        "spectral_contrast",
        "spectral_flatness",
        # "spectral_rolloff",
        # "poly_features",
        "tonnetz",
        "zero_crossing_rate",

        "tempogram",
        # "fourier_tempogram"
    ]
    answer = []
    for i in range(len(s), len(s) + 1):
        lists = findsubsets(s, i)
        for l in lists:
            print(l)
            res = 0.0
            for j in range(5):
                res += logistic_regression(dataset, l)
            res /= 5
            answer.append([l, res])
    answer = sorted(answer, key=lambda res: res[1], reverse=True)
    for i in answer:
        print(i[0])
        print(str(i[1]))