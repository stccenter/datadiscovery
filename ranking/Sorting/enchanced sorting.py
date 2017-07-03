"""
Created on Tue Jun  6 13:22:42 2017
@author: larakamal

enchanced sorting 
this code uses insertion sort to rank the documents 
the input of the code is a saved machine learning model 
from ranking_save_model.py and a data file from data/labelled_queries 
the output of the file is a ranked data file,
change the directory of the output file accordingly in line 67

"""

from keras.models import Sequential
from keras.layers import Dense
from keras.models import model_from_json
import pandas as pd
import numpy as np
from math import log10
import csv 
 
#sorting function 
def insertion_sort(features, model):
    #initialize rank from 0 to length of features
    rank = []
    for i in range(len(features)):
        rank.append(i)
        
    #insertion sort to sort features and rank list
    #insertion sort assumes transitivity 
    for i in range(1, len(features)):
            j = i-1
            temp = features[i]
            temp2 = rank[i]
            while(model.predict_classes((features[j] - temp).reshape(1,-1)) == 0 and (j >=0)):
                features[j+1] = features[j]
                rank[j+1] = rank[j]
                j = j - 1
                features[j+1] = temp
                rank[j+1] = temp2
    return rank      

#load model 
json_file = open('model.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
loaded_model = model_from_json(loaded_model_json)
loaded_model.load_weights("model.h5")

#compile model 
model.compile(optimizer='rmsprop',loss='binary_crossentropy', metrics=['accuracy'])

#extract 10 features from documents 
dataframe = pd.read_csv("data/labelled_queries/quikscat.csv")
features = dataframe.ix[:,0:10]
features = scaler.transform(features)

rank = insertion_sort(features, model)
            
#re-arrange documents accordingly
rows = dataframe.ix[:,0:11]
sorted_rows =[]
for i in rank:
    sorted_rows.append(rows.values[i])

#save file 
with open('results/test/quikscat_sorted1.csv', 'w', encoding = 'utf-8-sig') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['term_score', 'releaseDate_score', 'versionNum_score', 'processingL_score', 'allPop_score','monthPop_score', 'userPop_score', 'spatialR_score','temporalR_score','click_score','label'])
    for i in sorted_rows:
        writer.writerow(i)
        
