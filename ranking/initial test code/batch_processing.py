# The code here is only used to test whether DL can be used for ranking

#load training & test data
import csv
import numpy as np
from sklearn import preprocessing

train_data = []
train_labels = []
test_data = []
test_labels = []
with open('/Users/yjiang/Dropbox/DLData/humanlabelled.csv', newline='') as csvfile:
    spamreader = csv.reader(csvfile, delimiter=',', quotechar='|')
    i = 0;
    for row in spamreader:
        length = len(row)
        tmpfeature = row[0: length-1]
        feature = [float(x) for x in tmpfeature]
        label = int(row[length-1])
        if label == -1:
            label = 0
        i += 1
        var = i%3
        if var == 1:
            test_data.append(feature)
            test_labels.append(label)
        else:
            train_data.append(feature)
            train_labels.append(label)

scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data =  scaler.transform(test_data)

#train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Activation

#construct network
model = Sequential()
model.add(Dense(20, input_dim=10, activation='relu'))
model.add(Dense(20, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Train the model
model.fit(train_data, train_labels, epochs=40, batch_size=128)

# Evaluate
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
print(loss_and_metrics)