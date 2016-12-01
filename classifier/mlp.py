#!/data/opt/brew/bin/python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as skpp

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB

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

    train_data, test_data, train_labels, test_labels = train_test_split

    mlp = MLPClassifier()
    mlp.fit(train_data, train_labels)
    print mlp.score(test_data, test_labesl)


if __name__ == "__main__":
    main()
