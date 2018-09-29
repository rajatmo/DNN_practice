import glob

from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense
from keras.optimizers import SGD
import itertools as it
import numpy as np
from ROOT import TCanvas, TPaveLabel, TPaveText, TPavesText, TText
from ROOT import TArrow, TLine
from ROOT import gROOT, gBenchmark
import ROOT


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





#=======================List of variables to include

varlist=[
	# (medium b)
	#'htmiss',
	#'lep2_eta',
	#'mbb', 
	#'nBJetMedium',
	#'ptbb_loose',
	'avg_dr_jet', 
	'b1_loose_pt', 
	'b2_loose_pt', 
	'detabb_loose',
	'dr_lep1_tau_os',
	'dr_lep2_tau_ss',
	'dr_leps',
	'drbb_loose', 
	'lep1_conePt',
	'lep1_eta',
	'lep2_conePt',
	'max_lep_eta', 
	'mbb_loose', 
	'min_lep_eta', 
	'mindr_lep1_jet', 
	'mindr_lep2_jet', 
	'mindr_tau_jet',
	'mT_lep1',
	'mT_lep2',
	'mTauTauVis',
	'nBJetLoose',
	'nJet',
	'ptbb', 
	'ptmiss', 
	'tau_eta',
	'tau_pt', 
]



###============================path to root files

forBDTtraining="forBDTtraining"
channel='2los_1tau_Tight'
rootfilepath = "../root/forBDTtraining/*/*.root"
list = glob.glob(rootfilepath)
#========convert list of filenames to list of tfile objects

root_files=[]
for x in range(0,len(list)-1):
	try:	root_files.append(TFile.Open(list[x]))
	except: continue
#=================A dictionary to contain all data
##the list is: ['signal','ttH_hbb','TTW','TTZ','Rares','EWK'
myDict = {'signal':[],'ttH_hbb':[],'TTW':[],'TTZ':[],'Rares':[],'EWK':[]}
#====================Filling the dictionary

for categories in myDict.keys():
	try:
		print "\n\nEntering processes "+categories+"\nRoot files found in category: "
		for root_file in root_files:
			path_in = channel+"/sel/evtntuple/"+categories+"/evtTree"
			evt_tree=root_file.Get(path_in)
			if (evt_tree is not None) :
				if isinstance(evt_tree,ROOT.TTree):
					print categories+"\t"+root_file.GetName()
					myDict[categories].append(evt_tree)
	except:continue
#======================Setting up TMVA and Loading Data

dataloader = TMVA.DataLoader('dataset')
for branch in myDict['signal'][0].GetListOfBranches():
	for selection in varlist:
		if (selection == branch.GetName()):
			print selection
			dataloader.AddVariable(branch.GetName())

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')

dataloader.AddSignalTree(myDict['signal'][0], 1.0)

for categories in myDict.keys():
	if categories == 'signal': continue
	for i in range(0,(len(myDict[categories])-1)):
		if myDict[categories][i].GetEntries()>1:
			dataloader.AddBackgroundTree(myDict[categories][i], 1.0)


#====================building the model

model = Sequential()
model.add(Dense(50, activation='sigmoid', input_dim=22))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(50, activation='relu'))
model.add(Dense(2, activation='softmax'))


# ====================Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.03), metrics=['accuracy', ])

# =====================Store model to file
model.save('model.h5')
model.summary()

# =========================Book methods
#factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher',
#                   '!H:!V:Fisher:VarTransform=D,G')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras',
                   '!H:!V:FilenameModel=model.h5:NumEpochs=50')

# ===============Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()


