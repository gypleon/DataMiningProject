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
from sklearn.model_selection import train_test_split
from sklearn import metrics 
data_diabetes=pd.read_csv('disbetes_cleaned_python31129.csv');


#Caussian=0, AfricanAmerican=1, other=2

data_defined=data_diabetes.apply(lambda x: pd.factorize(x)[0])
#readmitted <30 1, not 30  0

#
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
data_defined2, targets = encode_target(data_defined, "readmitted")    
Y=data_defined2['Target'];
features = list(data_defined2.columns[:16])
#X=data_defined2[features]
#features_2= list(data_defined2.columns[3:11])
#X=data_defined2[features_2]

X=data_defined2[['age','admission_type_id','discharge_disposition_id','time_in_hospital','number_inpatient','A1Cresult','age']]

train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=.2, random_state=100)
tree = DecisionTreeClassifier(max_depth=5,random_state=0)
#dt = DecisionTreeClassifier(min_samples_split=5, random_state=5)
tree.fit(train_data, train_labels)
print('tree',tree.score(test_data, test_labels) )
result=tree.predict(test_data)



from sklearn.model_selection import cross_val_score
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


from sklearn.naive_bayes import MultinomialNB
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

mnb = MultinomialNB()
mnb.fit(train_data, train_labels)
print( "MNB", mnb.score(test_data, test_labels))


#MLPClassifier
mlp = MLPClassifier(hidden_layer_sizes=(8, 8), max_iter=300, verbose=True)
mlp.fit(train_data, train_labels)
print("MLP", mlp.score(test_data, test_labels))

rfc = RandomForestClassifier(criterion="entropy", max_features=None, n_jobs=-1, verbose=True)
rfc.fit(train_data, train_labels)
print("RFC", rfc.score(test_data, test_labels))

abc = AdaBoostClassifier()
abc.fit(train_data, train_labels)
print("ABC", abc.score(test_data, test_labels))


