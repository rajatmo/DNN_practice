
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import SGD
import itertools as it
import numpy as np

from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
import math , array 
import random

import glob
from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD

# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
'''
# Load data
if not isfile('tmva_class_example.root'):
    call(['curl', '-O', 'http://root.cern.ch/files/tmva_class_example.root'])
'''


varlist=[
			#'HadTop_eta', 
			#'HadTop_pt',
			'avg_dr_jet',
			#'dr_lep1_tau_os',
		 	'dr_lep2_tau_ss',
			#'dr_lepOS_HTfitted', 
			#'dr_lepOS_HTunfitted', 
			# 'dr_lepSS_HTfitted', 
			# 'dr_lepSS_HTunfitted',
			#'dr_leps', 
			#'dr_tau_HTfitted', #problematic
			# 'dr_tau_HTunfitted', #problematic
			#'evtWeight',
			#'fitHTptoHTmass', 
			#'fitHTptoHTpt', 
			#'genTopPt', #decrease auc
			# 'genWeight',
			#'htmiss', 
			# 'lep1_conePt', 
			# 'lep1_eta',
			# 'lep1_fake_prob', 
			# 'lep1_genLepPt',
			#'lep1_pt', 
			'lep1_tth_mva', 
			# 'lep2_conePt', 
			# 'lep2_eta', 
			#'lep2_fake_prob', 
			# 'lep2_genLepPt',
			#'lep2_pt', 
			'lep2_tth_mva', #
			# 'lumiScale',
			#'mT_lep1', 
			'mT_lep2', 
			'mTauTauVis', 
			#'mass_lepOS_HTfitted',
			#  'mass_lepSS_HTfitted',
			#'max_lep_eta', 
			#'mbb', 
			'mbb_loose',  
			#'b1_loose_pt', 
			#'b2_loose_pt', 
			#'drbb_loose', 
			#'detabb_loose',
			#'min_lep_eta',
			'mindr_lep1_jet', 
			'mindr_lep2_jet', 
			'mindr_tau_jet',
			#'mvaDiscr_2lss',
			#  'mvaOutput_2lss_ttV',
			#  'mvaOutput_2lss_ttbar',
			#  'mvaOutput_hadTopTagger',
			#'mvaOutput_hadTopTaggerWithKinFit', 
			# 'ptbb', 
			# 'ptbb_loose', 
			'ptmiss', 
			'tau_eta',
			#'tau_fake_prob', 
			# 'tau_genTauPt', 
			# 'tau_mva', 
			# 'lep2_charge', 
			# 'lep2_isTight',
			'tau_pt', 
			# 'unfittedHadTop_eta', 
			# 'unfittedHadTop_pt', 
			# #'bWj1Wj2_isGenMatched',
			#'bWj1Wj2_isGenMatchedWithKinFit',
			#  #'hadtruth', 
			# 'lep1_charge',
			#  'lep1_isTight',
			#'lep1_tau_charge', 
			# 'nBJetLoose',
			#  'nBJetMedium',
			'nJet', 
			# 'nLep',
			#  'nTau',
			#  #'tau_charge', 
			# 'tau_isTight', 
			# 'run',
			#  'lumi', 
			# 'evt'
		]


bdtType="evtLevelTT_TTH"
channelInTree="2los_1tau_Tight"



if bdtType=="evtLevelTT_TTH" : 
    keys=[
            'ttHJetToNonbb',
            'TTGJets',
            'TTJets_DiLept',
            'TTJets_SingleLeptFromT',
            'TTJets_SingleLeptFromTbar',

            ]
    
if bdtType=="evtLevelTTV_TTH" : 
    keys=[
            'TTWJetsToLNu_ext1',
            'TTWJetsToLNu_ext2',
            'TTWW',
            
            
            ]
    
inputPath= '/home/rajat/github/DNN_practice/root/forBDTtraining'



data = TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/ttHJetToNonbb_M125_amcatnlo_forBDTtraining_central_1.root')
signal = data.Get('/2los_1tau_Tight/sel/evtntuple/signal/evtTree')



data1 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/ttHJetTobb_M125/ttHJetTobb_M125_forBDTtraining_central_1.root')
background1 = data1.Get('/2los_1tau_Tight/sel/evtntuple/ttH_hbb/evtTree')

data2 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTJets_DiLept/TTJets_DiLept_forBDTtraining_central_1.root')
background2 = data2.Get('/2los_1tau_Tight/sel/evtntuple/TT/evtTree')

data3 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTJets_SingleLeptFromT/TTJets_SingleLeptFromT_forBDTtraining_central_1.root')
background3 = data3.Get('/2los_1tau_Tight/sel/evtntuple/TT/evtTree')


data4 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTJets_SingleLeptFromTbar/TTJets_SingleLeptFromTbar_forBDTtraining_central_1.root')
background4 = data4.Get('/2los_1tau_Tight/sel/evtntuple/TT/evtTree')

data5 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTWJetsToLNu_ext1/TTWJetsToLNu_ext1_forBDTtraining_central_1.root')
background5 = data5.Get('/2los_1tau_Tight/sel/evtntuple/TTW/evtTree')

data6 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTWJetsToLNu_ext2/TTWJetsToLNu_ext2_forBDTtraining_central_1.root')
background6 = data6.Get('/2los_1tau_Tight/sel/evtntuple/TTW/evtTree')

data7 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTZToLL_M10_ext1/TTZToLL_M10_ext1_forBDTtraining_central_1.root')
background7 = data7.Get('/2los_1tau_Tight/sel/evtntuple/TTZ/evtTree')

data8 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTZToLL_M10_ext2/TTZToLL_M10_ext2_forBDTtraining_central_1.root')
background8 = data8.Get('/2los_1tau_Tight/sel/evtntuple/TTZ/evtTree')

data9 =  TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/TTZToLL_M-1to10/TTZToLL_M-1to10_forBDTtraining_central_1.root')
background9 = data9.Get('/2los_1tau_Tight/sel/evtntuple/TTZ/evtTree')


dataloader = TMVA.DataLoader('dataset')
for branch in signal.GetListOfBranches():
    for selection in varlist:
        if (selection == branch.GetName()):
            dataloader.AddVariable(branch.GetName())

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background1, 1.0)
dataloader.AddBackgroundTree(background2, 1.0)
dataloader.AddBackgroundTree(background3, 1.0)
dataloader.AddBackgroundTree(background4, 1.0)
#dataloader.AddBackgroundTree(background5, 1.0)
#dataloader.AddBackgroundTree(background6, 1.0)
#dataloader.AddBackgroundTree(background7, 1.0)
#dataloader.AddBackgroundTree(background8, 1.0)
#dataloader.AddBackgroundTree(background9, 1.0)
#dataloader.AddBackgroundTree(background5, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=15000:nTrain_Background=5000:SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=14))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))


# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.03), metrics=['accuracy', ])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
#factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
#                   '!H:!V:Fisher:VarTransform=D,G')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   '!H:!V:FilenameModel=model.h5:NumEpochs=100')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()









