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
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from keras.models import Sequential
from keras.layers import Dense
from sorting import *
from evaluation import *  

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

# another hypothese: training_auto_new_query_filter.csv
with open('/Users/yjiang/Dropbox/DLData/auto_training.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_train_data = [data for data in data_iter]

# Load testing data (human-labelled)
with open('/Users/yjiang/Dropbox/DLData/humanlabelled.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_test_data = [data for data in data_iter]

train_data, train_labels = convert(input_train_data)
test_data_x, test_labels = convert(input_test_data)

# StandardScaler uses z-score, MinMaxScaler converts data to [0,1]
scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data =  scaler.transform(test_data_x)

classifiers = [linear_model.LogisticRegression(C=1e5), 
               SVC(kernel="linear", C=0.025),
               KNeighborsClassifier(20),
               DecisionTreeClassifier(max_depth=4)]
               #SVC(gamma=1, C=1)]

names = ["Logistic regression", "Linear SVM", "KNN", "Decision tree"] # "Non-linear SVM"]
shorts = ["lr", "svm", "knn", "tree"]

# =============================================================================
# for i in range(0, len(classifiers)):
#     model = classifiers[i].fit(train_data, train_labels)
#     predict = model.predict(test_data)
#     print("{}: {}".format(names[i], accuracy_score(test_labels, predict))) # or you can just do str(int);
#     get_search_results(model, scaler, shorts[i])
#     evaluate(shorts[i])
# =============================================================================
# great reference here: http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html


#construct network
model = Sequential()
model.add(Dense(25, input_dim=10, activation='relu'))
model.add(Dense(20, input_dim=10, activation='relu'))
model.add(Dense(15, activation='relu'))
model.add(Dense(10, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(5, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=80, batch_size=128)

# Evaluate
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
print(loss_and_metrics)

get_search_results(model, scaler, "dnn")
evaluate("dnn")

