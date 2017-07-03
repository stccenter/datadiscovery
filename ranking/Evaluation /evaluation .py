# -*- coding: utf-8 -*-
"""
Created on Tue May 30 13:13:44 2017
@author: larakamal

total evaluation evalutes the sorted documents 
using precision and NDCG
change the directory of the sorted documents from lines 79-87
change the directory of the precision and NDCG graphs from 
line 345 and 352
"""
from math import log10
import math
import csv 

def getNDCG(list, k):
    #convert to double 
   dcg = float(getDCG(list,k))
   idcg = float(getIDCG(list,k))
   ndcg = 0.0
   if (idcg > 0.0):
       ndcg = dcg/idcg
   return ndcg 


def getPrecision(list, k):
    size = len(list)
    if (size == 0 or k == 0):
        return 0.0
        
    if(k > size):
        k = size
    rel_doc_num = getRelevantDocNum(list,k)
    #convert to double 
    precision = float(float(rel_doc_num)/float(k))
    return precision 

      
def getRelevantDocNum(list,k):
    size = len(list)
    if (size == 0 or k == 0):
        return 0
    if (k > size):
        k = size
    rel_num = 0
    for i in range(k):
       if list[i] > 5:
           rel_num = rel_num + 1
    return rel_num 


def getDCG(list,k):
    size = len(list)
    if (size == 0 or k == 0):
        return 0.0
    if (k > size):
        k = size
    #convert to double
    dcg = list[0]
    dcg = float(dcg)

    for i in range(1,k):
        rel = list[i]
        pos = i+1
        rel_log = log10(pos)/log10(2)
        rel_log = float(rel_log)
        dcg = dcg + (rel/rel_log)
    return dcg 


def getIDCG(list, k):
    # sort list
    sortedList = list
    sortedList = sorted(sortedList, key=int, reverse=True)
    idcg = getDCG(sortedList, k)
    return float(idcg) 

#change directory of the ranked documents
dataframe1 = pd.read_csv("results/rank/gravity_sorted.csv")
dataframe2 = pd.read_csv("results/rank/ocean_pressure_sorted.csv")
dataframe3 = pd.read_csv("results/rank/ocean_temperature_sorted.csv")
dataframe4 = pd.read_csv("results/rank/ocean_wind_sorted.csv")
dataframe5 = pd.read_csv("results/rank/pathfinder_sorted.csv")
dataframe6 = pd.read_csv("results/rank/quikscat_sorted.csv")
dataframe7 = pd.read_csv("results/rank/radar_sorted.csv")
dataframe8 = pd.read_csv("results/rank/saline_density_sorted.csv")
dataframe9 = pd.read_csv("results/rank/sea_ice_sorted.csv")


label1 = dataframe1.ix[:,10:11]
label2 = dataframe2.ix[:,10:11]
label3 = dataframe3.ix[:,10:11]
label4 = dataframe4.ix[:,10:11]
label5 = dataframe5.ix[:,10:11]
label6 = dataframe6.ix[:,10:11]
label7 = dataframe7.ix[:,10:11]
label8 = dataframe8.ix[:,10:11]
label9 = dataframe9.ix[:,10:11]

temp_list1 = label1['label'].tolist()
temp_list2 = label2['label'].tolist()
temp_list3 = label3['label'].tolist()
temp_list4 = label4['label'].tolist()
temp_list5 = label5['label'].tolist()
temp_list6 = label6['label'].tolist()
temp_list7 = label7['label'].tolist()
temp_list8 = label8['label'].tolist()
temp_list9 = label9['label'].tolist()

label_list1 = [];
label_list2 = [];
label_list3 = [];
label_list4 = [];
label_list5 = [];
label_list6 = [];
label_list7 = [];
label_list8 = [];
label_list9 = [];


for i in range(len(temp_list1)):
    if temp_list1[i] == 'Excellent':
        label_list1.append(7) 
    elif temp_list1[i] == 'Very good':
        label_list1.append(6) 
    elif temp_list1[i] == 'Good':
        label_list1.append(5) 
    elif temp_list1[i] == 'Ok':
        label_list1.append(4) 
    elif temp_list1[i] == 'Bad':
        label_list1.append(3) 
    elif temp_list1[i] == 'Very bad':
        label_list1.append(2) 
    elif temp_list1[i] == 'Terrible':
        label_list1.append(1) 
    else:
        label_list1.append(0)
        
for i in range(len(temp_list2)):
    if temp_list2[i] == 'Excellent':
        label_list2.append(7) 
    elif temp_list2[i] == 'Very good':
        label_list2.append(6) 
    elif temp_list2[i] == 'Good':
        label_list2.append(5) 
    elif temp_list2[i] == 'Ok':
        label_list2.append(4) 
    elif temp_list2[i] == 'Bad':
        label_list2.append(3) 
    elif temp_list2[i] == 'Very bad':
        label_list2.append(2) 
    elif temp_list2[i] == 'Terrible':
        label_list2.append(1) 
    else:
        label_list2.append(0)

for i in range(len(temp_list3)):
    if temp_list3[i] == 'Excellent':
        label_list3.append(7) 
    elif temp_list3[i] == 'Very good':
        label_list3.append(6) 
    elif temp_list3[i] == 'Good':
        label_list3.append(5) 
    elif temp_list3[i] == 'Ok':
        label_list3.append(4) 
    elif temp_list3[i] == 'Bad':
        label_list3.append(3) 
    elif temp_list3[i] == 'Very bad':
        label_list3.append(2) 
    elif temp_list3[i] == 'Terrible':
        label_list3.append(1) 
    else:
        label_list3.append(0)
        
        
for i in range(len(temp_list4)):
    if temp_list4[i] == 'Excellent':
        label_list4.append(7) 
    elif temp_list4[i] == 'Very good':
        label_list4.append(6) 
    elif temp_list4[i] == 'Good':
        label_list4.append(5) 
    elif temp_list4[i] == 'Ok':
        label_list4.append(4) 
    elif temp_list4[i] == 'Bad':
        label_list4.append(3) 
    elif temp_list4[i] == 'Very bad':
        label_list4.append(2) 
    elif temp_list4[i] == 'Terrible':
        label_list4.append(1) 
    else:
        label_list4.append(0)
        
for i in range(len(temp_list5)):
    if temp_list5[i] == 'Excellent':
        label_list5.append(7) 
    elif temp_list5[i] == 'Very good':
        label_list5.append(6) 
    elif temp_list5[i] == 'Good':
        label_list5.append(5) 
    elif temp_list5[i] == 'Ok':
        label_list5.append(4) 
    elif temp_list5[i] == 'Bad':
        label_list5.append(3) 
    elif temp_list5[i] == 'Very bad':
        label_list5.append(2) 
    elif temp_list5[i] == 'Terrible':
        label_list5.append(1) 
    else:
        label_list5.append(0)
        
        
for i in range(len(temp_list6)):
    if temp_list6[i] == 'Excellent':
        label_list6.append(7) 
    elif temp_list6[i] == 'Very good':
        label_list6.append(6) 
    elif temp_list6[i] == 'Good':
        label_list6.append(5) 
    elif temp_list6[i] == 'Ok':
        label_list6.append(4) 
    elif temp_list6[i] == 'Bad':
        label_list6.append(3) 
    elif temp_list6[i] == 'Very bad':
        label_list6.append(2) 
    elif temp_list6[i] == 'Terrible':
        label_list6.append(1) 
    else:
        label_list6.append(0)
        
for i in range(len(temp_list7)):
    if temp_list7[i] == 'Excellent':
        label_list7.append(7) 
    elif temp_list7[i] == 'Very good':
        label_list7.append(6) 
    elif temp_list7[i] == 'Good':
        label_list7.append(5) 
    elif temp_list7[i] == 'Ok':
        label_list7.append(4) 
    elif temp_list7[i] == 'Bad':
        label_list7.append(3) 
    elif temp_list7[i] == 'Very bad':
        label_list7.append(2) 
    elif temp_list7[i] == 'Terrible':
        label_list7.append(1) 
    else:
        label_list7.append(0)
        
for i in range(len(temp_list8)):
    if temp_list8[i] == 'Excellent':
        label_list8.append(7) 
    elif temp_list8[i] == 'Very good':
        label_list8.append(6) 
    elif temp_list8[i] == 'Good':
        label_list8.append(5) 
    elif temp_list8[i] == 'Ok':
        label_list8.append(4) 
    elif temp_list8[i] == 'Bad':
        label_list8.append(3) 
    elif temp_list8[i] == 'Very bad':
        label_list8.append(2) 
    elif temp_list8[i] == 'Terrible':
        label_list8.append(1) 
    else:
        label_list8.append(0)
        
for i in range(len(temp_list9)):
    if temp_list9[i] == 'Excellent':
        label_list9.append(7) 
    elif temp_list9[i] == 'Very good':
        label_list9.append(6) 
    elif temp_list9[i] == 'Good':
        label_list9.append(5) 
    elif temp_list9[i] == 'Ok':
        label_list9.append(4) 
    elif temp_list9[i] == 'Bad':
        label_list9.append(3) 
    elif temp_list9[i] == 'Very bad':
        label_list9.append(2) 
    elif temp_list9[i] == 'Terrible':
        label_list9.append(1) 
    else:
        label_list9.append(0)

NDCG_list1 = [] 
NDCG_list2 = []   
NDCG_list3 = []
NDCG_list4 = []   
NDCG_list5 = []   
NDCG_list6 = []   
NDCG_list7 = []   
NDCG_list8 = []   
NDCG_list9 = []   
       
for i in range(1,41):
    k = i
    NDCG_list1.append(getNDCG(label_list1,k))
    NDCG_list2.append(getNDCG(label_list2,k))
    NDCG_list3.append(getNDCG(label_list3,k))
    NDCG_list4.append(getNDCG(label_list4,k))
    NDCG_list5.append(getNDCG(label_list5,k))
    NDCG_list6.append(getNDCG(label_list6,k))
    NDCG_list7.append(getNDCG(label_list7,k))
    NDCG_list8.append(getNDCG(label_list8,k))
    NDCG_list9.append(getNDCG(label_list9,k))
    
    
precision_list1 = []
precision_list2 = []
precision_list3 = []
precision_list4 = []
precision_list5 = []
precision_list6 = []
precision_list7 = []
precision_list8 = []
precision_list9 = []


for i in range(1,41):
    k = i
    precision_list1.append(getPrecision(label_list1,k))
    precision_list2.append(getPrecision(label_list2,k))
    precision_list3.append(getPrecision(label_list3,k))
    precision_list4.append(getPrecision(label_list4,k))
    precision_list5.append(getPrecision(label_list5,k))
    precision_list6.append(getPrecision(label_list6,k))
    precision_list7.append(getPrecision(label_list7,k))
    precision_list8.append(getPrecision(label_list8,k))
    precision_list9.append(getPrecision(label_list9,k))
    
    
    
total_list_NDCG = []
for i in range(len(NDCG_list1)):
    average = (NDCG_list1[i] + NDCG_list2[i]+ NDCG_list3[i] + NDCG_list4[i]+ NDCG_list5[i] + NDCG_list6[i] + NDCG_list7[i] + NDCG_list8[i] + NDCG_list9[i])/9
    array = np.array([NDCG_list1[i],NDCG_list2[i], NDCG_list3[i], NDCG_list4[i], NDCG_list5[i], NDCG_list6[i], NDCG_list7[i], NDCG_list8[i], NDCG_list9[i], average])
    total_list_NDCG.append(array)

total_list_precision = []
for i in range(len(precision_list1)):
    average = (precision_list1[i] + precision_list2[i]+ precision_list3[i] + precision_list4[i]+ precision_list5[i] + precision_list6[i] + precision_list7[i] + precision_list8[i] + precision_list9[i])/9
    array = np.array([precision_list1[i],precision_list2[i], precision_list3[i], precision_list4[i], precision_list5[i], precision_list6[i], precision_list7[i], precision_list8[i], precision_list9[i], average])
    total_list_precision.append(array)
   
with open('results/rank/NDCG_graph.csv', 'w', encoding = 'utf-8-sig') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['label'])
    writer.writerow(['gravity', 'ocean_pressure', 'ocean_temperature', 'ocean_wind', 'pathfinder','quikscat', 'radar', 'saline_density','sea_ice', 'MLP'])
    for i in total_list_NDCG:
        writer.writerow(i)
        
with open('results/rank/precision_graph.csv', 'w', encoding = 'utf-8-sig') as outcsv:
    writer = csv.writer(outcsv)
    writer.writerow(['label'])
    writer.writerow(['gravity', 'ocean_pressure', 'ocean_temperature', 'ocean_wind', 'pathfinder','quikscat', 'radar', 'saline_density','sea_ice', 'MLP'])
    for i in total_list_precision:
        writer.writerow(i) 