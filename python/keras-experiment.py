#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Aug 24 02:39:54 2018

@author: rajat
"""



from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import SGD
import itertools as it
import numpy as np
from sklearn.model_selection import train_test_split
from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
import math , array 
import random
from sklearn.metrics import roc_curve,auc

       
filenameS = '/home/rajat/github/DNN_practice/root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/ttHJetToNonbb_M125_amcatnlo_forBDTtraining_central_1.root'
filenameB = '~/github/DNN_practice/root/forBDTtraining/TTJets_DiLept/TTJets_DiLept_forBDTtraining_central_1.root'


arrS = root2array(filenameS, '/2los_1tau_Tight/sel/evtntuple/signal/evtTree')
arrB = root2array(filenameB, '/2los_1tau_Tight/sel/evtntuple/TT/evtTree')
arrS = np.array(arrS.tolist())
arrB = np.array(arrB.tolist())



arrD = np.append(arrS, arrB, axis = 0)


#arrS.view(np.recarray)


predicate= []

for i in arrS:
    predicate.append(1)
    
for i in arrB:
    predicate.append(0)

X_train, X_test, y_train, y_test = train_test_split(arrD, predicate, test_size=0.33, random_state=42)
#X_train, X_test, y_train, y_test = train_test_split(arrD, predicate, test_size=0.33, random_state=42)

#arrB.shape
dummy = y_test

#X_train = to_categorical(X_train)
y_train = to_categorical(y_train)

#X_test = to_categorical(X_test)
y_test = to_categorical(y_test)




'''
arrB = []
arrS = []
arrD = []
predicate= []
predicate_binary = []
'''
#=======================================================================


model = Sequential()

model.add(Dense(300,activation='relu',input_shape=(76,)))

model.add(Dense(300,activation='relu'))

model.add(Dense(300,activation='relu'))

model.add(Dense(300,activation='relu'))

model.add(Dense(300,activation='relu'))

model.add(Dense(300,activation='relu'))

model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer=SGD(),
              metrics=['accuracy'])


history= model.fit(X_train, 
          y_train, 
          epochs=10, 
          batch_size=100, 
          verbose = 1,
          validation_data=(X_test,y_test)
          )
y_score = []
for i in range(X_test.shape[0]):
    y_score.append( model.evaluate(X_test[:i+1], y_test[:i+1]))
#X_test[4:5]
#model.evaluate(X_test[i:i+1], y_test[i:i+1])
fpr = dict()
tpr = dict()
roc_auc = dict()
#y_test.shape[1]
#range(y_test.shape[1])
y_score = np.array(y_score)
for i in range(y_test.shape[1]):
    fpr[i], tpr[i], _ = roc_curve(y_test[:, i], y_score[:, i])
    roc_auc[i] = auc(fpr[i], tpr[i])
    
import matplotlib.pyplot as plt
plt.plot(tpr[1],fpr[1])
  
#roc_curve(y_test[:, 0], y_score[:, 0])

#type(y_test[:, 0])
#type( y_score[:, 0])
#(y_test[:, 0], y_score[:, 0])
#y_test[:, i]
#y_score[:, i]
#range(4)
'''
x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])

x.view(np.recarray)

predicatum_binary = to_categorical(predicatum)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(datam, predicatum_binary, epochs=5, batch_size=32)
'''