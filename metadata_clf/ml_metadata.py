import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
import cv2
import random
import pickle
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score
from sklearn.pipeline import Pipeline

from sklearn.neural_network import MLPClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC



def plot_coeff(clf, feat_names):
    coef = clf.coef_.ravel()
    top_pos = np.argsort(coef)[-5:]
    top_neg = np.argsort(coef)[:5]
    top_coeff = np.hstack([top_neg,top_pos])
    plt.bar(np.arange(10),coef=[top_coeff])
    plt.show()

# Reading in the data from CSV files
df1 = pd.read_csv('cancer_res_csv3.csv')
df2 = pd.read_csv('full_data.csv')


# Preparing for train_test_split
all_inputs_1 = df1[['skews_r_f','skews_r_fc','skews_g_f','skews_g_fc','skews_b_f','skews_b_fc','kurtosis_r_f','kurtosis_r_fc','kurtosis_g_f','kurtosis_g_fc','kurtosis_b_f','kurtosis_b_fc','mean_total_f_r','mean_total_f_g','mean_total_f_b','mean_total_fc_r','mean_total_fc_g','mean_total_fc_b']].values

all_classes_1 = df1['cancer'].values

all_inputs_2 = df2[['skews_r_f','skews_r_fc','skews_g_f','skews_g_fc','skews_b_f','skews_b_fc','kurtosis_r_f','kurtosis_r_fc','kurtosis_g_f','kurtosis_g_fc','kurtosis_b_f','kurtosis_b_fc','mean_total_f_r','mean_total_f_g','mean_total_f_b','mean_total_fc_r','mean_total_fc_g','mean_total_fc_b','mean_left_f_r','mean_left_f_g','mean_left_f_b','mean_left_fc_r','mean_left_fc_g','mean_left_fc_b','mean_right_f_r','mean_right_f_g','mean_right_f_b','mean_right_fc_r','mean_right_fc_g','mean_right_fc_b']].values
all_classes_2 = df2['cancer'].values

steps = [('SVM', SVC())]
pipeline = Pipeline(steps) # define the pipeline object.
parameters = {'SVM__C':[0.001,0.1,10,100,10e5], 'SVM__gamma':[0.1,0.01,0.001,0.0001]}
grid = GridSearchCV(pipeline, param_grid=parameters, cv=5)


# Splitting the data into training and testing sets with 70/30 ratio
(train_inputs_1, test_inputs_1, train_classes_1, test_classes_1) = train_test_split(all_inputs_1, all_classes_1, train_size=0.7)
(train_inputs_2, test_inputs_2, train_classes_2, test_classes_2) = train_test_split(all_inputs_2, all_classes_2, train_size=0.7)

# Using the data with Decision Tree Classifier
dtc_1 = DecisionTreeClassifier()
dtc_1.fit(train_inputs_1,train_classes_1)
res_1 = dtc_1.score(test_inputs_1,test_classes_1)
print("Decision Tree Classifier of csv1: ",res_1)

dtc_2 = DecisionTreeClassifier()
dtc_2.fit(train_inputs_2,train_classes_2)
res_2 = dtc_2.score(test_inputs_2,test_classes_2)
print("Decision Tree Classifier of csv2: ",res_2)

# Using the data with Linear SVC Models
svc_1 = SVC(kernel="linear",C=10,gamma=.0001, verbose=True)
svc_1.fit(train_inputs_1,train_classes_1)
res_3 = svc_1.predict(test_inputs_1)
print("Linear SVC with C=1 of csv3:", accuracy_score(test_classes_1,res_3))

svc_2 = SVC(kernel="linear",C=10,gamma=.0001)
svc_2.fit(train_inputs_2,train_classes_2)
res_4 = svc_2.predict(test_inputs_2)
print("Linear SVC with C=1 of csv2:", accuracy_score(test_classes_2,res_4))
print(type(svc_1))
# Using Multilevel Perceptron
mlp = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,2))
mlp.fit(train_inputs_1,train_classes_1)
res5 = mlp.predict(test_inputs_1)
print("Neural Network", accuracy_score(test_classes_1,res5))

mlp2 = MLPClassifier(solver='lbfgs', hidden_layer_sizes=(5,2))
mlp2.fit(train_inputs_2,train_classes_2)
res6 = mlp2.predict(test_inputs_2)
print("Neural Network", accuracy_score(test_classes_2,res6))
grid.fit(train_inputs_2, train_classes_2)
print(grid.score(test_inputs_2,test_classes_2))
print(grid.best_estimator_)

filename = 'final_model.sav'
pickle.dump(svc_1,open(filename, 'wb'))
