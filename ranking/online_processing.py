# Read csv & model selection
import csv
import numpy as np
from sklearn.model_selection import train_test_split
# Train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Activation
# Plot the eva results    
import matplotlib.pyplot as plt

# Load training & test data
input_data = []

def convert(data_array):
    feature_array = []
    label_array = []
    for d in data_array:
        features = [float(x) for x in d[0:len(d)-1]]
        label = int(d[len(d)-1])
        if label == -1:
             label = 0
        feature_array.append(features)
        label_array.append(label)
    return feature_array, label_array

# Please replace it with your local file (header removed)
with open('/Users/yjiang/Dropbox/inputDataForSVM.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_data = [data for data in data_iter]
        
train, test = train_test_split(input_data, test_size = 0.3)
train_data, train_labels = convert(train)
test_data, test_labels = convert(test)

# Build NN 
model = Sequential()
model.add(Dense(8, input_dim=10, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Batch processing
#==============================================================================
# model.fit(train_data, train_labels, epochs=20, batch_size=256)
# 
# loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=256)
# print(loss_and_metrics)
#==============================================================================

# Online learning
increment = 100
chunks_train_data = [train_data[x:x+increment] for x in range(0, len(train_data), increment)]
chunks_train_labels = [train_labels[x:x+increment] for x in range(0, len(train_labels), increment)]

test_accuracy = [];
for i, el in enumerate(chunks_train_data):
    print(i)
    train_loss_and_metrics = model.train_on_batch(el, chunks_train_labels[i])
    print(train_loss_and_metrics)
    test_loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
    print(test_loss_and_metrics)
    test_accuracy.append(test_loss_and_metrics[1])

plt.plot(test_accuracy)
plt.ylabel('Accuracy of test data')
plt.xlabel('Round number')
plt.show()


