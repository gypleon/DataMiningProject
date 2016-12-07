# -*- coding: utf-8 -*-

import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({'font.size': 26})
from itertools import cycle

from sklearn.preprocessing import label_binarize
from sklearn.preprocessing import OneHotEncoder
from sklearn.preprocessing import RobustScaler

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
readm_group = data_defined2.groupby("Target").get_group(1)
healthy_group = data_defined2.groupby("Target").get_group(0)
num_keep_size = min(readm_group.shape[0], healthy_group.shape[0])
df_balance = readm_group.sample(num_keep_size, replace=False)
df_balance = df_balance.append(healthy_group.sample(num_keep_size, replace=False))
df_balance = df_balance.sample(2*num_keep_size, replace=False)
# labels = data_defined2['Target'];
# features = list(data_defined2.columns[:16])
# X=data_defined2[features]
labels = df_balance['Target'];
features = list(df_balance.columns[:16])
X=df_balance[features]
#features_2= list(data_defined2.columns[3:11])
#X=data_defined2[features_2]

#X=data_defined2[['age','admission_type_id','discharge_disposition_id','time_in_hospital','number_inpatient','A1Cresult','age']]

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

# Switch for all classifier
if_all_cls = True

# Train
classifiers = list()
classifier_names = list()

max_depth = 2
crit = "gini"
dtree = DecisionTreeClassifier(criterion=crit,max_depth=max_depth)
dtree.fit(train_data, train_labels.ravel())
export_graphviz(dtree, max_depth=max_depth,feature_names=features,class_names=["non-readmitted", "readmitted"],leaves_parallel=True,rounded=True)
# export_graphviz(dtree, max_depth=max_depth,feature_names=features,rounded=True)
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

    rfc = RandomForestClassifier(criterion=crit, max_features=None, n_jobs=-1, verbose=False)
    rfc.fit(train_data, train_labels.ravel())
    classifiers.append(rfc) 
    classifier_names.append('Random Forest')

    abc = AdaBoostClassifier(dtree)
    abc.fit(train_data, train_labels.ravel())
    classifiers.append(abc) 
    classifier_names.append('AdaBoost')

# Evaluation
# decision tree
print "=======  TREE"
tree_test_score = dtree.score(test_data, test_labels.ravel())
tree_train_score = dtree.score(train_data, train_labels.ravel())
test_results = dtree.predict(test_data)
print "test_acc:", tree_test_score, ", train_acc:", tree_train_score
print "cross valid score", np.mean(cross_val_score(dtree, X, labels.ravel(), cv=20))
print 'confusion_matrix', metrics.confusion_matrix( test_labels.ravel(), test_results)
print 'f1_score', metrics.f1_score(test_results, test_labels.ravel())
# tn,fp,fn,tp=metrics.confusion_matrix(test_labels,result).ravel()
# print('tn,fp,fn,tp',tn,fp,fn,tp)
# print('matthews_corrcoef',metrics.matthews_corrcoef(result, test_labels))
# print('cohen_kappa_score',metrics.cohen_kappa_score(result, test_labels))
# precision, recall, thresholds = metrics.precision_recall_curve(test_labels,result)  
# print('precision, recall, thresholds ',precision, recall, thresholds )
if if_all_cls:
    # multinomial naive bayes
    print "=======  MNB"
    test_score   = mnb.score(test_data, test_labels.ravel())
    train_score  = mnb.score(train_data, train_labels.ravel())
    test_results = mnb.predict(test_data)
    print "test_acc:", test_score, ", train_acc:", train_score
    print "cross valid score", np.mean(cross_val_score(mnb, X, labels.ravel(), cv=20))
    print 'confusion_matrix', metrics.confusion_matrix( test_labels.ravel(), test_results)
    print 'f2_score', metrics.f1_score(test_results, test_labels.ravel())
    # multi layer perceptron
    print "=======  MLP"
    test_score   = mlp.score(test_data, test_labels.ravel())
    train_score  = mlp.score(train_data, train_labels.ravel())
    test_results = mlp.predict(test_data)
    print "test_acc:", test_score, ", train_acc:", train_score
    print "cross valid score", np.mean(cross_val_score(mlp, X, labels.ravel(), cv=20))
    print 'confusion_matrix', metrics.confusion_matrix( test_labels.ravel(), test_results)
    print 'f2_score', metrics.f1_score(test_results, test_labels.ravel())
    # random forest
    print "=======  RFC"
    test_score   = rfc.score(test_data, test_labels.ravel())
    train_score  = rfc.score(train_data, train_labels.ravel())
    test_results = rfc.predict(test_data)
    print "test_acc:", test_score, ", train_acc:", train_score
    print "cross valid score", np.mean(cross_val_score(rfc, X, labels.ravel(), cv=20))
    print 'confusion_matrix', metrics.confusion_matrix( test_labels.ravel(), test_results)
    print 'f2_score', metrics.f1_score(test_results, test_labels.ravel())
    # adaptive boosting
    print "=======  ABC"
    test_score   = abc.score(test_data, test_labels.ravel())
    train_score  = abc.score(train_data, train_labels.ravel())
    test_results = abc.predict(test_data)
    print "test_acc:", test_score, ", train_acc:", train_score
    print "cross valid score", np.mean(cross_val_score(abc, X, labels.ravel(), cv=20))
    print 'confusion_matrix', metrics.confusion_matrix( test_labels.ravel(), test_results)
    print 'f2_score', metrics.f1_score(test_results, test_labels.ravel())

# Functions for Plot
def compute_roc_auc(classifiers, classifier_names, test_labels, test_data):
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    num_classes = test_labels.shape[1]
    ind = 0
    for classifier in classifiers:
        # all classifier adopt predict_proba() for roc_curve()
        # if hasattr(classifier, "decision_function"):
        if False:
            # print "########## desicion function ###########", classifier_names[ind]
            test_score = classifier.decision_function(test_data)
            # Compute ROC curve and ROC area for each class
            # print classifier_names[ind], test_score
            fpr[classifier_names[ind]], tpr[classifier_names[ind]], _ = metrics.roc_curve(test_labels.ravel(), test_score)
        else:
            # print "########## predict proba ###########", classifier_names[ind]
            test_score = classifier.predict_proba(test_data)
            # Compute ROC curve and ROC area for each class
            # print classifier_names[ind], test_score
            fpr[classifier_names[ind]], tpr[classifier_names[ind]], _ = metrics.roc_curve(test_labels.ravel(), test_score[:,1])
        roc_auc[classifier_names[ind]] = metrics.auc(fpr[classifier_names[ind]], tpr[classifier_names[ind]])
        ind += 1
    return fpr, tpr, roc_auc

def plot_roc(classifier_names, fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    ind = 0
    for classifier, color in zip(classifier_names, colors):
        plt.plot(fpr[classifier], tpr[classifier], color=color,
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

# Compute Precision-Recall and plot curve
def compute_precision_recall(classifiers, classifier_names, test_labels, test_data):
    precision = dict()
    recall = dict()
    average_precision = dict()
    num_classes = test_labels.shape[1]
    ind = 0
    for classifier in classifiers:
        test_score = classifier.predict_proba(test_data)
        precision[classifier_names[ind]], recall[classifier_names[ind]], _ = metrics.precision_recall_curve(test_labels.ravel(), test_score[:, 1])
        average_precision[classifier_names[ind]] = metrics.average_precision_score(test_labels.ravel(), test_score[:, 1])
        ind += 1
    return precision, recall, average_precision

# Plot Precision-Recall curve for each class
def plot_precision_recall_classes(classifier_names, precision, recall, average_precision):
    plt.clf()
    lw = 2
    colors = cycle(['navy', 'turquoise', 'darkorange', 'cornflowerblue', 'teal'])
    ind = 0
    for classifier, color in zip(classifier_names, colors):
        plt.plot(recall[classifier_names[ind]], precision[classifier_names[ind]], color=color, lw=lw,
                 label='{0} (area = {1:0.2f})'
                       ''.format(classifier_names[ind], average_precision[classifier_names[ind]]))
        ind += 1
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall')
    plt.ylabel('Precision')
    plt.title('Precision-Recall Curve for Classifiers')
    plt.legend(loc="upper right")
    plt.show()

# Plot
# Receiver Operating Characteristic

# Plot dtree
# fpr, tpr, roc_auc = compute_roc_auc([dtree], ["Decision Tree"], test_labels, test_data)
# plot_roc(["Decision Tree"], fpr, tpr, roc_auc)

# Plot all classifiers
fpr, tpr, roc_auc = compute_roc_auc(classifiers, classifier_names, test_labels, test_data)
plot_roc(classifier_names, fpr, tpr, roc_auc)

# Precision-Recall
precision, recall, average_precision = compute_precision_recall(classifiers, classifier_names, test_labels, test_data)
plot_precision_recall_classes(classifier_names, precision, recall, average_precision)
