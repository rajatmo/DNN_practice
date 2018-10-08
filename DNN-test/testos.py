import numpy as np
import pandas as pd
import glob

#importing KERAS
import keras
from keras.layers import Embedding, Input, Flatten, Dense, Activation
from keras.models import Model, Sequential
from keras.optimizers import SGD
from keras.utils import to_categorical
from keras.wrappers.scikit_learn import KerasClassifier


from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import auc


import matplotlib
from matplotlib import pyplot as plt



#=================================data import

pathwithwildcard = '../csv/2017/forBDTtraining/*/*.csv'
paths = glob.glob(pathwithwildcard)
df1_signal_dict = {}
df1_TT_dict = {}
df1_TTV_dict = {}

for path in paths:
    if 'ttHToNonbb' in path:
        df1_signal_dict[path] = pd.read_csv(path)
for path in paths:
    if not (('ttHToNonbb' in path) or ('PSweights' in path)):
        if 'TTTo' in path:
            df1_TT_dict[path] = pd.read_csv(path)
        elif 'TTW' in path or 'TTZ' in path:
            df1_TTV_dict[path] = pd.read_csv(path)
        else:
            continue


tmp_sig = []
for keys in df1_signal_dict.keys():
    tmp_sig.append(df1_signal_dict[keys])
tmp_TT = []
for keys in df1_TT_dict.keys():
    tmp_TT.append(df1_TT_dict[keys])
df1_signal = pd.concat(tmp_sig)
df1_TT = pd.concat(tmp_TT)
print df1_signal.shape, df1_TT.shape


varlist=[
    #'AK12_lead_mass',
    #'AK12_lead_pt',
    #'DR_AK12_tau',
    'avg_dr_jet',
    'b1_loose_pt',
    'b1_pt',
    'b2_loose_pt',
    'b2_pt',
    #'detabb',
    #'detabb_loose',
    'dr_lep1_tau_os',
    'dr_leps',
    'drbb',
    'drbb_loose',
    'htmiss',
    'lep1_conePt',
    'lep1_eta',
    'lep1_fake_prob',
    'lep1_pt',
    'lep2_conePt',
    'lep2_eta',
    'lep2_genLepPt',
    'lep2_pt',
    'mT_lep1',
    'mT_lep2',
    'mTauTauVis',
    'max_lep_eta',
    'mbb',
    'mbb_loose',
    'minDR_AK12_L',
    'minDR_AK12_lep',
    'minDR_HTTv2_L',
    'minDR_HTTv2_lep',
    'minDR_HTTv2_tau',
    'mindr_lep1_jet',
    'mindr_lep2_jet',
    'mindr_tau_jet',
    'ptbb',
    'ptbb_loose',
    'ptmiss',
    'tau_eta',
    'tau_mva',
    'tau_pt',
    'N_jetAK12',
    'nBJetLoose',
    'nJet',
    'nMuon'
]




#=========================data preprocessing
df1_signal = df1_signal[varlist]
df1_signal_target = [0]*df1_signal.shape[0]
df1_TT = df1_TT[varlist]
df1_TT_target = [1]*df1_TT.shape[0]
tmp0 = [df1_signal,df1_TT]
df1 = pd.concat(tmp0)
target = df1_signal_target+df1_TT_target
#df1['target'] = target

#train_test split
#df1_train = df1.sample(frac=.2)
#df1_test = df1.drop(df1_train.index)
df1_train, df1_test, df1_train_target, df1_test_target = train_test_split(df1, target, test_size=0.5)

#df1_train_target = df1_train['target'].copy()
#df1_train = df1_train.drop('target',axis=1)
#df1_test_target = df1_test['target'].copy()
#df1_test = df1_test.drop('target',axis=1)

df1_train_target_cat = to_categorical(df1_train_target)
df1_test_target_cat = to_categorical(df1_test_target)

print str(df1_train_target_cat.shape), df1_train.shape, df1_test.shape, df1_test_target_cat.shape

print df1_train.columns.values

#===============================defining keras model
model = Sequential()
descrip = [10,10,10,2]
for iterator in range(0,len(descrip)):
    if iterator == 0:
        model.add(Dense(descrip[iterator],activation='relu',input_dim=len(varlist)))
        continue
    elif iterator == len(descrip)-1:
        model.add(Dense(descrip[iterator],activation='softmax'))
        break
    else:
        model.add(Dense(descrip[iterator],activation='relu'))



model.summary()

model.compile(loss='binary_crossentropy',optimizer='adam',metrics=['binary_accuracy'])

num_epochs = 100
sizeofepoch = 10

hist = model.fit(df1_train,df1_train_target_cat,
                 batch_size=sizeofepoch,
                 epochs=num_epochs,
                 validation_data=(df1_test,df1_test_target_cat),
                 verbose = 1
                 )


#===================================roc curve

fpr_keras, tpr_keras, thresholds_keras = roc_curve(df1_test_target_cat[:,0],model.predict(df1_test)[:,0])
fpr_keras_train, tpr_keras_train, thresholds_keras_train = roc_curve(df1_train_target_cat[:,0],model.predict(df1_train)[:,0])
auc_keras = auc(fpr_keras, tpr_keras)
auc_keras_train = auc(fpr_keras_train, tpr_keras_train)
plt.figure(1)
#plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_keras, tpr_keras, label='pyKeras_test (area = {:.3f})'.format(auc_keras))
plt.plot(fpr_keras_train, tpr_keras_train, label='pyKeras_train (area = {:.3f})'.format(auc_keras_train))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.savefig('pyKeras_batch_size'+str(sizeofepoch)+'_epochs'+str(num_epochs)+'__'+str(len(varlist))+'vars'+'__depth'+str(len(descrip))+'.pdf')
plt.show()


