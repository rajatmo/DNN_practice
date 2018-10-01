import glob
import keras.optimizers
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


def train_DNN(bkg, path,varlist,frac,pykeras,GBDT,DNN_TMVA,AdaBDT):
	###============================path to root files

	forBDTtraining="forBDTtraining"
	channel='2los_1tau_Tight'
	rootfilepath = path+"*/*.root"
	list = glob.glob(rootfilepath)
	#========convert list of filenames to list of tfile objects

	root_files=[]
	for x in range(0,len(list)):
		try:	root_files.append(TFile.Open(list[x]))
		except: continue
	#=================A dictionary to contain all data
	##the list is: ['signal','ttH_hbb','TTW','TTZ','Rares','EWK']
	if bkg == "TTV":
		myDict = {
		'signal':[],
		'ttH_hbb':[],
		'TTW':[],
		'TTZ':[],
		#'Rares':[],
		#'EWK':[],
		#'TT':[]
		}
	if bkg == "bb":
		myDict = {
		'signal':[],
		'ttH_hbb':[],
		#'TTW':[],
		#'TTZ':[],
		#'Rares':[],
		#'EWK':[],
		#'TT':[]
		}
	elif bkg == "TT":
		myDict = {
		'signal':[],
		'ttH_hbb':[],
		#'TTW':[],
		#'TTZ':[],
		#'Rares':[],
		#'EWK':[],
		'TT':[]
		}
	elif bkg == "all":
		myDict = {
		'signal':[],
		'ttH_hbb':[],
		'TTW':[],
		'TTZ':[],
		'Rares':[],
		'EWK':[],
		'TT':[]
		}
	#====================Filling the dictionary
	nEvtTot = 0
	for categories in myDict.keys():
		try:
			print "\n\nEntering processes "+categories+"\nRoot files found in category: "
			for root_file in root_files:
				path_in = channel+"/sel/evtntuple/"+categories+"/evtTree"
				evt_tree=root_file.Get(path_in)
				if (evt_tree is not None) :
					if isinstance(evt_tree,ROOT.TTree):
						print categories+"\t"+root_file.GetName()+' '+str(evt_tree.GetEntries())+' entries.'
						myDict[categories].append(evt_tree)
						nEvtTot += evt_tree.GetEntries()
		except:
			continue
	#======================Setting up TMVA and Loading Data

	dataloader = TMVA.DataLoader('dataset')
	for branch in myDict['signal'][0].GetListOfBranches():
		for selection in varlist:
			if (selection == branch.GetName()):
				try:	dataloader.AddVariable(selection)
				except:
					print "\n\n\n\n\n\n\n"+selection+" has problems."

	TMVA.Tools.Instance()
	TMVA.PyMethodBase.PyInitialize()

	output = TFile.Open('TMVA.root', 'RECREATE')
	dataloader.AddSignalTree(myDict['signal'][0], 1.0)


	for categories in myDict.keys():
		if categories == 'signal': continue
		for i in range(0,(len(myDict[categories]))):
			if myDict[categories][i].GetEntries()!=0:
				dataloader.AddBackgroundTree(myDict[categories][i], 1.0)
	train_sig = myDict["signal"][0].GetEntries()*frac
	train_bkg = (nEvtTot - myDict["signal"][0].GetEntries())*frac
	dataloader.PrepareTrainingAndTestTree(TCut(''),'nTrain_Signal='+str(train_sig)+':nTrain_Background='+str(train_bkg)+':SplitMode=Random:NormMode=NumEvents:!V')
	
	# ==================================Book methods
	factory = TMVA.Factory('TMVAClassification', output,
	                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')
	if pykeras != 0:
		####=============PyKeras 			*

		#====================building the model
		model = Sequential()
		num_layers=100
		num_nodes=100
		model.add(Dense(num_nodes, activation='tanh', input_dim=len(varlist)))
		for i in range(0,num_layers):
			model.add(Dense(num_nodes, activation='tanh'))
		model.add(Dense(2, activation='relu'))
		# ====================Set loss and optimizer
		model.compile(loss='mean_squared_error',
		              optimizer=keras.optimizers.Adam(lr=0.1, beta_1=0.9, beta_2=0.999, epsilon=None, decay=0.0, amsgrad=False), metrics=['binary_accuracy', ])
		# =====================Store model to file
		model.save('model.h5')
		model.summary()

		factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras','!H:!V:FilenameModel=model.h5:NumEpochs=100')


	if GBDT!=0:	
		#####=============GBDT
		factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDTG','!H:!V:NTrees=1000::BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5')

	if DNN_TMVA!=0:
		#####=============DNN 				*
		preq='!H:V:ErrorStrategy=CROSSENTROPY:VarTransform=N:WeightInitialization=XAVIERUNIFORM:'
		layoutString='Layout=TANH|128,TANH|128,TANH|128,LINEAR:'
		layer0='TrainingStrategy=LearningRate=1e-1,Momentum=0.9,Repetitions=1,ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.5+0.5+0.5, Multithreading=True|'
		layer1='TrainingStrategy=LearningRate=1e-2,Momentum=0.9,Repetitions=1,ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.5+0.5+0.5, Multithreading=True|'
		layer2='LearningRate=1e-3,Momentum=0.9,Repetitions=1,ConvergenceSteps=20,BatchSize=256,TestRepetitions=10,WeightDecay=1e-4,Regularization=L2,DropConfig=0.0+0.5+0.5+0.5, Multithreading=True:'
		arch='Architecture=STANDARD'
		trainingString=preq+layoutString+layer0+layer1+layer2+arch
		

		try:	factory.BookMethod(dataloader, TMVA.Types.kDNN, 'DNN CPU',trainingString)
		except: print "kDNN is not working yet."

	if AdaBDT!=0:
		#####=============AdaBDT
		factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDT','!H:!V:NTrees=1000:nEventsMin=400:MaxDepth=3:BoostType=AdaBoost:SeparationType=GiniIndex:nCuts=20:PruneMethod=NoPruning')
	
	# ===============Run training, test and evaluation

	try:
		factory.TrainAllMethods()
		factory.TestAllMethods()
		factory.EvaluateAllMethods()
	except: print "Problem in training"
#End train_DNN














varlist=[
	'taupt',
	'avg_dr_jet',
	'njet',
	'tau_eta',
	'dr_leps',
	'mindr_tau_jet',
	'mindr_lep1_jet',
	'dr_lep1_tau_jet',
	'mTauTauVis',
	'mT_lep2',
	'mindr_lep2_jet',
	'dr_lep2_tau_ss',
	'dr_lep1_tau_os',
	'mbb_loose',
	'lep2_conePt',
	'nBJetLoose',
	'min_lep_eta',
	'mT_lep1',
	'ptmiss',
	'max_lep_eta',
	'lep1_conePt',
	'drbb_loose'
]

train_DNN('TTV','../root/forBDTtraining/',varlist,0.5,
	pykeras=1,GBDT=0,DNN_TMVA=0,AdaBDT=0)
