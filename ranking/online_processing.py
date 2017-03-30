# Read csv & model selection
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import plot_model
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
with open('/Users/yjiang/Dropbox/DLData/humanlabelled.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_data = [data for data in data_iter]
        
train, test = train_test_split(input_data, test_size = 0.3)
train_data, train_labels = convert(train)
test_data, test_labels = convert(test)

scaler = preprocessing.StandardScaler().fit(train_data)
train_data = scaler.transform(train_data)
test_data =  scaler.transform(test_data)

# Build NN 
model = Sequential()
model.add(Dense(8, input_dim=10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Batch processing
model.fit(train_data, train_labels, epochs=2, batch_size = 128)
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size = 128)
print(loss_and_metrics)
# Plot out the model shape
plot_model(model, to_file='model.png', show_shapes = "true")

# Online learning
#==============================================================================
# increment = 128
# chunks_train_data = [train_data[x:x+increment] for x in range(0, len(train_data), increment)]
# chunks_train_labels = [train_labels[x:x+increment] for x in range(0, len(train_labels), increment)]
# 
# test_accuracy = [];
# test_loss = [];
# train_accuracy = [];
# train_loss = [];
# for i, el in enumerate(chunks_train_data):
#     print(i)
#     train_loss_and_metrics = model.train_on_batch(el, chunks_train_labels[i])
#     print(train_loss_and_metrics)
#     train_loss.append(train_loss_and_metrics[0])
#     train_accuracy.append(train_loss_and_metrics[1])
#     test_loss_and_metrics = model.evaluate(test_data, test_labels, batch_size=128)
#     print(test_loss_and_metrics)
#     test_loss.append(test_loss_and_metrics[0])
#     test_accuracy.append(test_loss_and_metrics[1])
# 
# fig = plt.figure()
# 
# ax1 = fig.add_subplot(211)
# ax1.plot(train_loss)
# ax1.plot(test_loss)
# ax1.set_ylabel('Loss')
# ax1.set_xlabel('Round number')
# ax1.legend(['training', 'testing'], loc='upper left')
# 
# ax2 = fig.add_subplot(212)
# ax2.plot(train_accuracy)
# ax2.plot(test_accuracy)
# ax2.set_ylabel('Accuracy')
# ax2.set_xlabel('Round number')
# ax2.legend(['training', 'testing'], loc='upper left')
# plt.show()
#==============================================================================


