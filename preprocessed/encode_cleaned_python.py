#!/data/opt/brew/bin/python

import sys

import pandas as pd
import numpy as np

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.neural_network import MLPClassifier

PATH_CLEANED="./cleaned_python.csv"
# PATH_CLEANED="../dataset_diabetes/diabetic_data.csv"
PATH_ENCODED="./encoded_python.csv"

def preprocess_col(val, i):
# readmitted
    if 16 == i:
        if val == "readmitted":
            return val
        elif val == "Not30":
            return 0
        elif val == "<30":
            return 1
        else:
            print "Abnormal Value in:", i, ", Val:", val
# diabetesMed
    elif 15 == i:
        if val == "diabetesMed":
            return val
        elif val == "No":
            return 0
        if val == "Yes":
            return 1
        else:
            print "Abnormal Value in:", i, ", Val:", val
# change
    elif 14 == i:
        if val == "change":
            return val
        elif val == "No":
            return 0
        elif val == "Ch":
            return 1
        else:
            print "Abnormal Value in:", i, ", Val:", val
# insulin
    elif 13 == i:
        if val == "insulin":
            return val
        elif val == "No":
            return 0
        elif val == "Down":
            return 1
        elif val == "Steady":
            return 2
        elif val == "Up":
            return 3
        else:
            print "Abnormal Value in:", i, ", Val:", val
# glyburide
    elif 12 == i:
        if val == "glyburide":
            return val
        elif val == "No":
            return 0
        elif val == "Down":
            return 1
        elif val == "Steady":
            return 2
        elif val == "Up":
            return 3
        else:
            print "Abnormal Value in:", i, ", Val:", val
# glipizide
    elif 11 == i:
        if val == "glipizide":
            return val
        elif val == "No":
            return 0
        elif val == "Down":
            return 1
        elif val == "Steady":
            return 2
        elif val == "Up":
            return 3
        else:
            print "Abnormal Value in:", i, ", Val:", val
# A1Cresult
    elif 10 == i:
        if val == "A1Cresult":
            return val
        elif val == "None":
            return 0
        elif val == "Norm":
            return 1
        elif val == ">7":
            return 2
        elif val == ">8":
            return 3
        else:
            print "Abnormal Value in:", i, ", Val:", val
# number_inpatient
    elif 9 == i:
        if val == "number_inpatient":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# number_emergency
    elif 8 == i:
        if val == "number_emergency":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# number_outpatient
    elif 7 == i:
        if val == "number_outpatient":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# time_in_hospital
    elif 6 == i:
        if val == "time_in_hospital":
            return val
        elif val == "[1-4]":
            return 0
        elif val == "[5-8]":
            return 1
        elif val == "[9-14]":
            return 2
        else:
            print "Abnormal Value in:", i, ", Val:", val
# admission_source_id
    elif 5 == i:
        if val == "admission_source_id":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# discharge_disposition_id
    elif 4 == i:
        if val == "discharge_disposition_id":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# Admission Type ID
    elif 3 == i:
        if val == "admission_type_id":
            return val
        elif type(int(val)) == type(1):
            return int(val)
        else:
            print "Abnormal Value in:", i, ", Val:", val
# Age
    elif 2 == i:
        if val == "age":
            return val
        elif val == "[0-40)":
            return 0
        elif val == "[40-70)":
            return 1
        elif val == "[70-100)":
            return 2
        else:
            print "Abnormal Value in:", i, ", Val:", val
# Gender
    elif 1 == i:
        if val == "gender":
            return val
        elif val == "Male":
            return 0
        elif val == "Female":
            return 1
        else:
            print "Abnormal Value in:", i, ", Val:", val
# Race
    elif 0 == i:
        if val == "race":
            return val
        elif val == "AfricanAmerican":
            return 0
        elif val == "Asian":
            return 1
        elif val == "Caucasian":
            return 2
        elif val == "Hispanic":
            return 3
        elif val == "Other":
            return 4
        else:
            print "Abnormal Value in:", i, ", Val:", val
        

def main():
# encode
    df = pd.read_csv(PATH_CLEANED, header=None, low_memory=False)
    for i in range(17):
        df[i] = df[i].apply(preprocess_col, args=(i,))
    # special case
    df[0][0] = 2
    df.to_csv(PATH_ENCODED, header=False, index=False)

if __name__ == "__main__":
    main()
