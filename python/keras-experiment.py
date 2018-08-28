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
from sklearn.metrics import roc_curve
       
filenameS = '/home/rajat/~/Git/clones/dummy-repo/root/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_forBDTtraining_central_1.root'
filenameB = '/home/rajat/~/Git/clones/dummy-repo/root/TTWJets_LO/TTWJets_LO_forBDTtraining_central_1.root'


arrS = root2array(filenameS, '/2los_1tau_forBDTtraining/sel/evtntuple/signal/evtTree')
arrB = root2array(filenameB, '/2los_1tau_forBDTtraining/sel/evtntuple/TTW/evtTree')

arrD = np.append(arrS, arrB, axis = 0)


#arrS.view(np.recarray)

arrS = np.array(arrS.tolist())
arrB = np.array(arrB.tolist())


predicate= []

for i in arrS:
    predicate.append(1)
    
for i in arrB:
    predicate.append(0)

X_train, X_test, y_train, y_test = train_test_split(arrD, predicate, test_size=0.33, random_state=42)

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

model.add(Dense(10,activation='sigmoid',input_shape=(72,)))

#model.add(Dense(10,activation='sigmoid'))

model.add(Dense(2,activation='softmax'))

model.summary()

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])


history= model.fit(X_train, 
          y_train, 
          epochs=100, 
          batch_size=5000, 
          verbose = 1,
          validation_data=(X_test,y_test)
          )



score = model.evaluate(X_test, y_test)

y_pred_keras=model.predict(X_train)#.ravel()

fpr_keras, tpr_keras, thresholds_keras = roc_curve(dummy, y_pred_keras)


'''
x = np.array([(1.0, 2), (3.0, 4)], dtype=[('x', float), ('y', int)])

x.view(np.recarray)

predicatum_binary = to_categorical(predicatum)

model.compile(loss='categorical_crossentropy',
              optimizer='sgd',
              metrics=['accuracy'])

model.fit(datam, predicatum_binary, epochs=5, batch_size=32)
'''