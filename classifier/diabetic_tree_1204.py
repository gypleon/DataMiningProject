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

data_diabetes=pd.read_csv('disbetes_cleaned_python31129.csv');


#Caussian=0, AfricanAmerican=1, other=2

data_defined=data_diabetes.apply(lambda x: pd.factorize(x)[0])




def encode_target(df, target_column):
    """Add column to df with integers for the target.

    Args
    ----
    df -- pandas DataFrame.
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
    

data_defined2, targets = encode_target(data_defined, "readmitted")    
Y=data_defined2['Target'];
features = list(data_defined2.columns[:16])
X=data_defined2[features]




train_data, test_data, train_labels, test_labels = train_test_split(X, Y, test_size=.2, random_state=66)
tree = DecisionTreeClassifier()
#dt = DecisionTreeClassifier(min_samples_split=5, random_state=5)
tree.fit(train_data, train_labels)
print('tree'), tree.score(test_data, test_labels)

from sklearn.model_selection import cross_val_score
cross_val_score(tree, train_data, train_labels, cv=10)
cross_val_score(tree, test_data, test_labels, cv=10)



#
#with open("iris.dot", 'w') as f: f = export_graphviz(kk, out_file=f)
#
#import os
#
#import  pydotplus
##dot_data = export_graphviz(kk, out_file=None) 
##graph = pydot.graph_from_dot_data(dot_data) 
##graph.write_pdf("iris.pdf") 
#
#
#from IPython.display import Image  
#dot_data = export_graphviz(kk, out_file=None)  
#
#graph = pydotplus.graph_from_dot_data(dot_data)  
#
#Image(graph.create_png())  
##
##
#def visualize_tree(tree, feature_names):
#    """Create tree png using graphviz.
#
#    Args
#    ----
#    tree -- scikit-learn DecsisionTree.
#    feature_names -- list of feature names.
#    """
#    with open("dt.dot", 'w') as f:
#        export_graphviz(tree, out_file=f,
#                        feature_names=feature_names)
#
#    command = ["dot", "-Tpng", "dt.dot", "-o", "dt.png"]
#    try:
#        subprocess.check_call(command)
#    except:
#        exit("Could not run dot, ie graphviz, to "
#             "produce visualization")
#        
#             
#visualize_tree(dt, features)