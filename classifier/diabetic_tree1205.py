# -*- coding: utf-8 -*-
"""
Created on Mon Dec  5 20:29:53 2016

@author: Samuel_lab
"""

# -*- coding: utf-8 -*-
"""
Created on Wed Nov 30 17:00:25 2016

@author: Samuel_lab
"""
import os
import subprocess

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

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
labels=data_defined2['Target'];
num_classes = len(classes)
features = list(data_defined2.columns[:16])
#X=data_defined2[features]
#features_2= list(data_defined2.columns[3:11])
#X=data_defined2[features_2]

X=data_defined2[['age','admission_type_id','discharge_disposition_id','time_in_hospital','number_inpatient','A1Cresult','age']]

# Convert into np.array
X = np.array(X)
labels = np.array(labels)
train_data, test_data, train_labels, test_labels = train_test_split(X, labels, test_size=.2, random_state=100)

# Train
tree = DecisionTreeClassifier(max_depth=5,random_state=0)
print type(tree)
what = tree.fit(train_data, train_labels)
print type(what)
exit()

mnb = MultinomialNB()
mnb.fit(train_data, train_labels)

mlp = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=300, verbose=False)
mlp.fit(train_data, train_labels)

rfc = RandomForestClassifier(criterion="entropy", max_features=None, n_jobs=-1, verbose=False)
rfc.fit(train_data, train_labels)

abc = AdaBoostClassifier()
abc.fit(train_data, train_labels)

# Evaluation
# decision tree
tree_score = tree.score(test_data, test_labels)
print('tree', tree_score)
tree_score = what.decision_funcion()
result=tree.predict(test_data)
print('Cross_value_train')
print(np.mean(cross_val_score(tree, train_data, train_labels, cv=20)))
print('Cross_value_test')
print(np.mean((cross_val_score(tree, test_data, test_labels, cv=20))))
print('confusion_matrix',metrics.confusion_matrix( result, test_labels ))
tn,fp,fn,tp=metrics.confusion_matrix(test_labels,result).ravel()
print('tn,fp,fn,tp',tn,fp,fn,tp)
print('matthews_corrcoef',metrics.matthews_corrcoef(result, test_labels))
print('cohen_kappa_score',metrics.cohen_kappa_score(result, test_labels))
print('f1_score',metrics.f1_score(result, test_labels)  )
precision, recall, thresholds = metrics.precision_recall_curve(test_labels,result)  
print('precision, recall, thresholds ',precision, recall, thresholds )
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
def compute_roc_auc(test_labels, test_score):
    # Compute ROC curve and ROC area for each class
    fpr = dict()
    tpr = dict()
    roc_auc = dict()
    for i in range(num_classes):
        fpr[i], tpr[i], _ = metrics.roc_curve(test_labels[:, i], test_score[:, i])
        roc_auc[i] = metrics.auc(fpr[i], tpr[i])
    # Compute micro-average ROC curve and ROC area
    fpr["micro"], tpr["micro"], _ = roc_curve(y_test.ravel(), y_score.ravel())
    roc_auc["micro"] = auc(fpr["micro"], tpr["micro"])
    return fpr, tpr, roc_auc

def plot_roc(fpr, tpr, roc_auc):
    plt.figure()
    lw = 2
    plt.plot(fpr[2], tpr[2], color='darkorange',
             lw=lw, label='ROC curve (area = %0.2f)' % roc_auc[2])
    plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver operating characteristic example')
    plt.legend(loc="lower right")
    plt.show()

# plot tree
fpr, tpr, roc_auc = compute_roc_auc(test_labels, tree_score)
plot_roc(fpr, tpr, roc_auc)



# Precision-Recall
