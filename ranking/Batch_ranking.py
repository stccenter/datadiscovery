
# coding: utf-8

# In[ ]:

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

chunks_train_data = [train_data[x:x+100] for x in range(0, len(train_data), 100)]
chunks_train_labels = [train_labels[x:x+100] for x in range(0, len(train_labels), 100)]

for i, el in enumerate(chunks_train_data):
    print(i)
    train_loss_and_metrics = model.train_on_batch(el, chunks_train_labels[i])
    print(train_loss_and_metrics)
    test_loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
    print(test_loss_and_metrics)


# In[ ]:



