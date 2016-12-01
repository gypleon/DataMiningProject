#!/data/opt/brew/bin/python

import numpy as np

from sklearn import preprocessing as skpp

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
    array = preprocessing(dataset)
    print array


if __name__ == "__main__":
    main()
