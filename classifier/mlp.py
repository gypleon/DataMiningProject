#!/data/opt/brew/bin/python

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as skpp

from sklearn.neural_network import MLPClassifier

PATH_DATASET="../preprocessed/encoded_python_features.csv"
PATH_LABELS="../preprocessed/encoded_python_labels.csv"

def load_dataset(path):
    dataset = np.loadtxt(path, delimiter=",")
    return dataset

def preprocessing(dataset):
    enc = skpp.OneHotEncoder()
    enc.fit(dataset)
    return enc.transform(dataset).toarray()

def main():
    dataset = load_dataset(PATH_DATASET)
    labels = load_dataset(PATH_DATASET)
    dataset = preprocessing(dataset)

    train_data = dataset[:18000]
    train_labels = labels[:18000]
    test_data = dataset[18001:]
    test_labels = labels[18001:]

    mlp = MLPClassifier()
    mlp.fit(train_data, train_labels)
    print mlp.score(test_data, test_labesl)


if __name__ == "__main__":
    main()
