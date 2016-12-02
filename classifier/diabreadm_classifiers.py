#!/data/opt/brew/bin/python

import numpy as np

# import matplotlib.pyplot as plt
# from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as skpp

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import MultinomialNB

PATH_DATASET="../preprocessed/encoded_python_features.csv"
PATH_LABELS="../preprocessed/encoded_python_labels.csv"

def load_dataset(path):
    dataset = np.loadtxt(path, delimiter=",")
    return dataset

def preprocessing(dataset):
    enc = skpp.OneHotEncoder()
    enc.fit(dataset)
    dataset = enc.transform(dataset).toarray()
    # dataset = StandardScaler().fit_transform(dataset)
    return dataset

def main():
    dataset = load_dataset(PATH_DATASET)
    labels = load_dataset(PATH_LABELS)

    dataset = preprocessing(dataset)

    train_data, test_data, train_labels, test_labels = train_test_split(dataset, labels, test_size=.2, random_state=66)

    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=300, verbose=True)
    mlp.fit(train_data, train_labels)
    print "MLP", mlp.score(test_data, test_labels)

    rfc = RandomForestClassifier(criterion="entropy", max_features=None, n_jobs=-1, verbose=True)
    rfc.fit(train_data, train_labels)
    print "RFC", rfc.score(test_data, test_labels)

    abc = AdaBoostClassifier()
    abc.fit(train_data, train_labels)
    print "ABC", abc.score(test_data, test_labels)

    mnb = MultinomialNB()
    mnb.fit(train_data, train_labels)
    print "MNB", mnb.score(test_data, test_labels)



if __name__ == "__main__":
    main()
