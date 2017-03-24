# coding: utf-8
#load training & test data
import csv
import numpy as np

train_data = []
train_labels = []
test_data = []
test_labels = []

with open('D:/pythone workspace/ranking/inputDataForSVM.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0;
    for row in spamreader:
        i += 1
        var = i%3
        if var == 1:
            length = len(row)
            test_data.append(row[0: length-1])
            test_labels.append(row[length-1])
        else:
            length = len(row)
            train_data.append(row[0: length-1])
            train_labels.append(row[length-1])

#train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Activation

#construct network
model = Sequential()
model.add(Dense(20, input_dim=10, ,activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='mse',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=10, batch_size=128)

# Evaluate
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
print(loss_and_metrics)
