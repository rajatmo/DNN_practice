
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
			# 'lep1_tth_mva', 
			# 'lep2_conePt', 
			# 'lep2_eta', 
			#'lep2_fake_prob', 
			# 'lep2_genLepPt',
			#'lep2_pt', 
			# 'lep2_tth_mva', #
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
if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTTo2L2Nu_PSweights','TTToHadronic','TTToHadronic_PSweights','TTToSemiLeptonic','TTToSemiLeptonic_PSweights']
if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTWJets','TTZJets']
inputPath= '/home/rajat/github/DNN_practice/root/forBDTtraining/'
for folderName in keys :
        print (folderName, channelInTree)
        if 'TTT' in folderName :
                sampleName='TT'
                target=0
        if folderName=='ttHToNonbb' :
                sampleName='signal'
                target=1
        if 'TTW' in folderName :
                sampleName='TTW'
                target=0
        if 'TTZ' in folderName :
                sampleName='TTZ'
                target=0
        inputTree = channelInTree+'/sel/evtntuple/'+sampleName+'/evtTree'
        if folderName=='ttHToNonbb' :
            procP1=glob.glob(inputPath+"/"+folderName+"_M125_powheg/"+folderName+"*.root")
            list=procP1
        elif ('TTT' in folderName):
            procP1=glob.glob(inputPath+"/"+folderName+"/"+folderName+"*.root")
            list=procP1
        elif ('TTW' in folderName) or ('TTZ' in folderName):
            procP1=glob.glob(inputPath+"/"+folderName+"_LO/"+folderName+"*.root")
            list=procP1
        for ii in range(0, len(list)) :
            try: data = TFile.Open(list[ii])
            except : continue
            try: tree = data.Get(inputTree)
            except : continue
            if tree is not None :
				print(list(ii),inputTree)



data = TFile.Open('/home/rajat/github/DNN_practice/root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/ttHJetToNonbb_M125_amcatnlo_forBDTtraining_central_1.root')
signal = data.Get('/2los_1tau_Tight/sel/evtntuple/signal/evtTree')
data1 =  TFile.Open('~/github/DNN_practice/root/forBDTtraining/'+'TTWJetsToLNu_ext1/TTWJetsToLNu_ext1_forBDTtraining_central_1.root')
background = data1.Get('/2los_1tau_Tight/sel/evtntuple/TTW/evtTree')





dataloader = TMVA.DataLoader('dataset')
for branch in signal.GetListOfBranches():
    for selection in varlist:
        if (selection == branch.GetName()):
            dataloader.AddVariable(branch.GetName())

dataloader.AddSignalTree(signal, 1.0)
for process in backgrounds:
	dataloader.AddBackgroundTree(backgrounds, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=20000:nTrain_Background=3000:SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=12))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(50, activation='sigmoid'))
model.add(Dense(2, activation='relu'))


# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
#factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
#                   '!H:!V:Fisher:VarTransform=D,G')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   '!H:!V:FilenameModel=model.h5:NumEpochs=10000:BatchSize=200')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()














