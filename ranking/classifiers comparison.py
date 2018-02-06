#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Tue Feb  6 11:11:33 2018
Compared muliple common supervised machine learning models
@author: yjiang
"""

import csv
from sklearn import preprocessing
from sklearn import linear_model
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC

# Load training & test data
input_test_data = []
input_train_data = []

def convert(data_array):
    feature_array = []
    label_array = []
    for d in data_array:
        features = [float(x) for x in d[0:len(d)-1]]
        label = int(d[len(d)-1])
        if(label==-1):
            label = 0
        feature_array.append(features)
        label_array.append(label)
    return feature_array, label_array

# Please replace it with your local file (header removed)
# Load traning data (user-generated)
with open('/Users/yjiang/Dropbox/DLData/auto_training.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_train_data = [data for data in data_iter]

# Load testing data (human-labelled)
with open('/Users/yjiang/Dropbox/DLData/humanlabelled.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_test_data = [data for data in data_iter]

        
# train, test = train_test_split(input_data, test_size = 0.3)
# test_data, test_labels = convert(test)
train_data, train_labels = convert(input_train_data)
test_data_x, test_labels = convert(input_test_data)

# scaler = preprocessing.StandardScaler().fit(test_data)
scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data =  scaler.transform(test_data_x)

classifiers = [linear_model.LogisticRegression(C=1e5), 
               SVC(kernel="linear", C=0.025),
               SVC(gamma=1, C=1)]

names = ["Logistic regression", "Linear regression", "Non-linear logistic regression"]

for i in range(0, len(classifiers)):
    model = classifiers[i].fit(train_data, train_labels)
    predict = model.predict(test_data)
    print("{}: {}".format(names[i], accuracy_score(test_labels, predict))) # or you can just do str(int);

# great reference here: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html
