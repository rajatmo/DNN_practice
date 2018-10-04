#!/usr/bin/env python

from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile

from keras.models import Sequential
from keras.layers import Dense, Activation
from keras.regularizers import l2
from keras.optimizers import SGD


varlist=[
    'AK12_lead_mass',
    'AK12_lead_pt',
    'DR_AK12_tau',
    'avg_dr_jet',
    'b1_loose_pt',
    'b1_pt',
    'b2_loose_pt',
    'b2_pt',
    'detabb',
    'detabb_loose',
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
    #'min_lep_eta',#problematic
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
    'nMuon',
    #'nTau',
]









# Setup TMVA
TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:Transformations=D,G:AnalysisType=Classification')


data = TFile.Open('../root/2017/forBDTtraining/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_forBDTtraining_central_1.root')
data1 = TFile.Open(' ../root/2017/forBDTtraining/TTWJets_LO/TTWJets_LO_forBDTtraining_central_1.root')
signal = data.Get('2los_1tau_forBDTtraining/sel/evtntuple/signal/evtTree')
background = data1.Get('2los_1tau_forBDTtraining/sel/evtntuple/TTW/evtTree')

dataloader = TMVA.DataLoader('dataset')
for branch in signal.GetListOfBranches():
	for variabs in varlist:
		if (branch.GetName()==variabs):
			dataloader.AddVariable(branch.GetName())

dataloader.AddSignalTree(signal, 1.0)
dataloader.AddBackgroundTree(background, 1.0)
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=4000:nTrain_Background=1500:SplitMode=Random:NormMode=NumEvents:!V')

# Generate model

# Define model
model = Sequential()
model.add(Dense(1000, activation='relu', W_regularizer=l2(1e-5), input_dim=len(varlist)))
#model.add(Dense(100, activation='relu'))
#model.add(Dense(10, activation='relu'))
model.add(Dense(2, activation='softmax'))

# Set loss and optimizer
model.compile(loss='categorical_crossentropy',
              optimizer=SGD(lr=0.01), metrics=['accuracy', ])

# Store model to file
model.save('model.h5')
model.summary()

# Book methods
factory.BookMethod(dataloader, TMVA.Types.kFisher, 'Fisher','!H:!V:Fisher:VarTransform=D,G')
factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras','H:!V:VarTransform=D,G:FilenameModel=model.h5:NumEpochs=40:BatchSize=40')

# Run training, test and evaluation
factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()
