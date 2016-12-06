#!/data/opt/brew/bin/python

import numpy as np

import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap

from sklearn.model_selection import train_test_split

from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing as skpp

from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import tree
from sklearn.naive_bayes import MultinomialNB

from sklearn.model_selection import cross_val_score
from sklearn import metrics

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

    dt = DecisionTreeClassifier()
    dt.fit(train_data, train_labels)
    print "Tree", dt.score(test_data, test_labels)
    # cross_val_score(dt, train_data, train_labels, cv=10)
    # tree.export_graphviz(dt, out_file="tree.dot", max_depth=5, class_names=["Non-Readmission", "Readmission"])
    print metrics.confusion_matrix(test_labels, dt.predict(test_data), labels=[0, 1])
    exit()

    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=300, verbose=False)
    mlp.fit(train_data, train_labels)
    print "Multi Layer Perceptron", mlp.score(test_data, test_labels)

    rfc = RandomForestClassifier(criterion="entropy", max_features=None, n_jobs=-1, verbose=False)
    rfc.fit(train_data, train_labels)
    print "Random Forest", rfc.score(test_data, test_labels)

    abc = AdaBoostClassifier()
    abc.fit(train_data, train_labels)
    print "AdaBoost", abc.score(test_data, test_labels)

    mnb = MultinomialNB()
    mnb.fit(train_data, train_labels)
    print "Nultinomial Naive Bayes", mnb.score(test_data, test_labels)



if __name__ == "__main__":
    main()
