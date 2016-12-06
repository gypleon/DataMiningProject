# -*- coding: utf-8 -*-

import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder

from sklearn.tree import DecisionTreeClassifier, export_graphviz
from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

from sklearn import metrics 
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score

data_diabetes=pd.read_csv('../preprocessed/disbetes_cleaned_python31129.csv');


#Caussian=0, AfricanAmerican=1, other=2

data_defined=data_diabetes.apply(lambda x: pd.factorize(x)[0])
#readmitted <30 1, not 30  0

#readm_group = data_defined.groupby('readmitted').get_group(1)
#nonreadm_group = data_defined.groupby('readmitted').get_group(0)
##num_keep_size = min(readm_group.shape[0], nonreadm_group.shape[0])
#num_keep_size = min(readm_group.shape[0], nonreadm_group.shape[0])
#resampled = readm_group.sample(num_keep_size, replace=False)
#resampled = resampled.append(nonreadm_group.sample(1*num_keep_size, replace=False))
#resampled = resampled.sample(2*num_keep_size, replace=False)


def encode_target(df, target_column):
    """Add column to df with integers for the target.
    Args  ----    df -- pandas DataFrame.
    target_column -- column to map to int, producing
                     new Target column.
    Returns
    -------
    df_mod -- modified DataFrame.
    targets -- list of target names.
    """
    df_mod = df.copy()
    targets = df_mod[target_column].unique()
    map_to_int = {name: n for n, name in enumerate(targets)}
    df_mod["Target"] = df_mod[target_column].replace(map_to_int)

    return (df_mod, targets)
    
#data_defined2, targets = encode_target(resampled, "readmitted")  
data_defined2, classes = encode_target(data_defined, "readmitted")    
labels = data_defined2['Target'];
features = list(data_defined2.columns[:16])
#X=data_defined2[features]
#features_2= list(data_defined2.columns[3:11])
#X=data_defined2[features_2]

X=data_defined2[['age','admission_type_id','discharge_disposition_id','time_in_hospital','number_inpatient','A1Cresult','age']]

# Convert into np.ndarray
X = np.array(X)
labels = np.array(labels)

# Binarize the output
# print X, X.shape, labels, labels.shape
labels = label_binarize(labels, classes = [0,1])
num_classes = labels.shape[1]
# print X, X.shape, labels, labels.shape
# exit()

# Generate Training & Testing Data
train_data, test_data, train_labels, test_labels = train_test_split(X, labels, test_size=.2, random_state=100)
print train_data, train_labels
print train_data.shape, train_labels.shape

# Switch for all classifier
if_all_cls = True

# Train
classifiers = list()
classifier_names = list()

dtree = DecisionTreeClassifier(max_depth=5,random_state=0)
dtree.fit(train_data, train_labels.ravel())
classifiers.append(dtree) 
classifier_names.append('Decision Tree')

if if_all_cls:

    mnb = MultinomialNB()
    mnb.fit(train_data, train_labels.ravel())
    classifiers.append(mnb) 
    classifier_names.append('Multinomial Naive Bayes')

    mlp = MLPClassifier(hidden_layer_sizes=(16, 16), max_iter=300, verbose=False)
    mlp.fit(train_data, train_labels.ravel())
    classifiers.append(mlp) 
    classifier_names.append('Multi Layer Perceptron')

    rfc = RandomForestClassifier(criterion="gini", max_features=None, n_jobs=-1, verbose=False)
    rfc.fit(train_data, train_labels.ravel())
    classifiers.append(rfc) 
    classifier_names.append('Random Forest')

    abc = AdaBoostClassifier()
    abc.fit(train_data, train_labels.ravel())
    classifiers.append(abc) 
    classifier_names.append('AdaBoost')

# Evaluation
# decision tree
tree_score = dtree.score(test_data, test_labels)
# print('dtree', tree_score)
# result=dtree.predict(test_data)
# cross validation needs label 1d-array
# print('Cross_value_train')
# print(np.mean(cross_val_score(dtree, train_data, train_labels.ravel(), cv=20)))
# print('Cross_value_test')
# print(np.mean((cross_val_score(dtree, test_data, test_labels.ravel(), cv=20))))
# print('confusion_matrix',metrics.confusion_matrix( result, test_labels ))
# tn,fp,fn,tp=metrics.confusion_matrix(test_labels,result).ravel()
# print('tn,fp,fn,tp',tn,fp,fn,tp)
# print('matthews_corrcoef',metrics.matthews_corrcoef(result, test_labels))
# print('cohen_kappa_score',metrics.cohen_kappa_score(result, test_labels))
# print('f1_score',metrics.f1_score(result, test_labels)  )
# precision, recall, thresholds = metrics.precision_recall_curve(test_labels,result)  
# print('precision, recall, thresholds ',precision, recall, thresholds )
if not if_all_cls:
    # multinomial naive bayes
    print("MNB", mnb.score(test_data, test_labels))
    # multi layer perceptron
    print("MLP", mlp.score(test_data, test_labels))
    # random forest
    print("RFC", rfc.score(test_data, test_labels))
    # adaptive boosting
    print("ABC", abc.score(test_data, test_labels))

# Plot
# Receiver Operating Characteristic
def compute_roc_auc(classifiers, classifier_names, test_labels, test_data):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = test_labels.shape[1]
    ind = 0
    for classifier in classifiers:
        if hasattr(classifier, "decision_function"):
            print "########## desicion function ###########", classifier_names[ind]
            test_score = classifier.decision_function(test_data)
            # Compute ROC curve and ROC area for each class
            print classifier_names[ind], test_score
            fpr[classifier_names[ind]], tpr[classifier_names[ind]], _ = metrics.roc_curve(test_labels.ravel(), test_score)
        else:
            print "########## predict proba ###########", classifier_names[ind]
            test_score = classifier.predict_proba(test_data)
            # Compute ROC curve and ROC area for each class
            print classifier_names[ind], test_score
            fpr[classifier_names[ind]], tpr[classifier_names[ind]], _ = metrics.roc_curve(test_labels.ravel(), test_score[:,1])
        roc_auc[classifier_names[ind]] = metrics.auc(fpr[classifier_names[ind]], tpr[classifier_names[ind]])
        ind += 1
    return fpr, tpr, roc_auc

def plot_roc(classifier_names, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    colors = ['b','g','r','c','m','y','k']
    ind = 0
    for classifier in classifier_names:
        plt.plot(fpr[classifier], tpr[classifier], color=colors[ind],
                 linewidth=lw, label='%s (area = %0.2f)' % (classifier_names[ind], roc_auc[classifier]))
        ind += 1
    plt.plot([0, 1], [0, 1], color='navy', linewidth=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

# plot dtree
# fpr, tpr, roc_auc = compute_roc_auc([dtree], ["Decision Tree"], test_labels, test_data)
# plot_roc(["Decision Tree"], fpr, tpr, roc_auc)
# plot all classifiers
fpr, tpr, roc_auc = compute_roc_auc(classifiers, classifier_names, test_labels, test_data)
plot_roc(classifier_names, fpr, tpr, roc_auc)



# Precision-Recall
