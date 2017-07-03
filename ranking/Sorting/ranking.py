"""
Created on Tue May 30 13:13:44 2017
@author: larakamal

This code is exactly like search ranking.ipynb
I added few lines to save the model to the current
directory from lines 93-98

"""

import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.model_selection import train_test_split
import os 

dataframe = pd.read_csv("data/auto_training_for_ipy.csv")
train_features = dataframe.ix[:,0:10]
train_labels = dataframe.ix[:,10:11]
train_labels['label'] = train_labels['label'].astype(int)
train_labels['label'] = train_labels['label'].map({-1: 0, 1: 1})
print('features: ', train_features.shape)
print('labels: ', train_labels.shape)
train_features.head()
train_features.describe()
(train_features.term_score.plot.line(lw=0.8))
plt.title('Term score')
plt.xlabel('ID')
train_features.term_score.hist()
print('mean: ', train_features.mean()[0])
print('var :', train_features.var()[0])

import seaborn as sns
correlations = train_features.corr()
corr_heat = sns.heatmap(correlations)
plt.title('Ranking feature Correlations')
(correlations
     .term_score
     .drop('term_score') 
     .sort_values(ascending=False)
     .plot
     .barh())

dataframe = pd.read_csv("data/humanlabelled_for_ipy.csv")
test_features = dataframe.ix[:,0:10]
test_labels = dataframe.ix[:,10:11]
test_labels['label'] = test_labels['label'].astype(int)
test_labels['label'] = test_labels['label'].map({-1: 0, 1: 1})

from sklearn import preprocessing
#Train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation
from keras.utils import plot_model
from keras.utils import np_utils

scaler = preprocessing.StandardScaler().fit(train_features)
train_features = scaler.transform(train_features)
test_features =  scaler.transform(test_features)
train_labels = np_utils.to_categorical(train_labels, num_classes=2)
test_labels = np_utils.to_categorical(test_labels, num_classes=2)
 
model = Sequential()

model.add(Dense(8, input_dim=10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(4, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])
              
test_accuracy = [];
test_loss = [];
train_accuracy = [];
train_loss = [];
  
increment = 64
chunks_train_data = [train_features[x:x+increment] for x in range(0, len(train_features), increment)]
chunks_train_labels = [train_labels[x:x+increment] for x in range(0, len(train_features), increment)]

for epoch in range(0, 5):
    for i, el in enumerate(chunks_train_data):
        #print(i)
        train_loss_and_metrics = model.train_on_batch(el, chunks_train_labels[i])
        #print(train_loss_and_metrics)
        train_loss.append(train_loss_and_metrics[0])
        train_accuracy.append(train_loss_and_metrics[1])
        test_loss_and_metrics = model.evaluate(test_features, test_labels, batch_size=128)
        #print(test_loss_and_metrics)
        test_loss.append(test_loss_and_metrics[0])
        test_accuracy.append(test_loss_and_metrics[1])
  
#save model   
model_json = model.to_json()
with open("model.json", "w") as json_file:
    json_file.write(model_json)
model.save_weights("model.h5")
print("\n saved model to disk")
        
fig = plt.figure()
#visualize the learning curve  
ax1 = fig.add_subplot(211)
ax1.plot(train_loss)
ax1.plot(test_loss)
ax1.set_ylabel('Loss')
ax1.set_xlabel('Iteration')
ax1.legend(['training', 'testing'], loc='upper left')

ax2 = fig.add_subplot(212)
ax2.plot(train_accuracy)
ax2.plot(test_accuracy)
ax2.set_ylabel('Accuracy')
ax2.set_xlabel('Iteration')
ax2.legend(['training', 'testing'], loc='upper left')

