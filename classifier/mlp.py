#!/data/opt/brew/bin/python

import numpy as np

from sklearn import preprocessing

PATH_DATASET="../preprocessed/encoded_python.csv"

def load_dataset(path):
    dataset = np.loadtxt(path, delimiter=",")
    print dataset

def main():
    dataset = load_dataset(PATH_DATASET)



if __name__ == "__main__":
    main()
