# The code here is only used to test the possibility of online learning where only humanlabelled data is used.


# Read csv & model selection
import csv
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
# Train data with DL model
from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.utils import np_utils
from keras.utils import plot_model
# Plot the eva results    
import matplotlib.pyplot as plt

# %reset

# Load training & test data
input_data = []

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
with open('/Users/yjiang/Dropbox/DLData/humanlabelled.csv', newline='') as csvfile:
    data_iter = csv.reader(csvfile, delimiter=',', quotechar='|')
    input_data = [data for data in data_iter]
        
train, test = train_test_split(input_data, test_size = 0.3)
train_data_x, train_labels_y = convert(train)
test_data, test_labels = convert(test)

# Preprocessing
scaler = preprocessing.StandardScaler().fit(train_data_x)
train_data = scaler.transform(train_data_x)
test_data =  scaler.transform(test_data)
train_labels = np_utils.to_categorical(train_labels_y, num_classes=2)
test_labels = np_utils.to_categorical(test_labels, num_classes=2)

# Build NN 
model = Sequential()
model.add(Dense(8, input_dim=10, activation='relu'))
model.add(Dense(7, activation='relu'))
model.add(Dense(6, activation='relu'))
model.add(Dense(2, activation='softmax'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Batch processing
model.fit(train_data, train_labels, epochs=100, batch_size = 128)
loss_and_metrics = model.evaluate(test_data, test_labels, batch_size = 128)
print(loss_and_metrics)

# Visualize weights
#==============================================================================
# weights = model.layers[0].get_weights()
# w0 = weights[0][0][0]
# w1 = weights[1][0]
#==============================================================================


# Plot out the model shape
#==============================================================================
# plot_model(model, to_file='model.png', show_shapes = "true")
#==============================================================================

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
# ax1.set_xlabel('Iteration')
# ax1.legend(['training', 'testing'], loc='upper left')
# 
# ax2 = fig.add_subplot(212)
# ax2.plot(train_accuracy)
# ax2.plot(test_accuracy)
# ax2.set_ylabel('Accuracy')
# ax2.set_xlabel('Iteration')
# ax2.legend(['training', 'testing'], loc='upper left')
# plt.show()
#==============================================================================

#==============================================================================
# output = np.column_stack((train_data_x, train_labels_y))
# np.savetxt("/Users/yjiang/Dropbox/DLData/train_result_1.csv", output, delimiter=",", 
#             fmt="%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%d")
#==============================================================================

