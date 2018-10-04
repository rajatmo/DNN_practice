import ROOT

import glob
import itertools as it
import numpy as np
import math , array 
import random
from subprocess import call
from os.path import isfile
from numpy.lib.recfunctions import append_fields
from itertools import product

import keras
import keras.optimizers
from keras.optimizers import SGD
from keras.models import Sequential
from keras.utils import to_categorical
from keras.layers import Dense, Activation
from keras.regularizers import l2

from ROOT import TCanvas, TMVA, TFile, TTree, TCut


bkg='TTV'
path='../root/2017/forBDTtraining/'
frac=0.5

varlist=[
    'AK12_lead_mass',
    'AK12_lead_pt',
    'DR_AK12_tau',
    #'HTT_boosted',
    #'HTT_boosted_WithKinFit',
    #'HTT_semi_boosted',
    #'HTT_semi_boosted_WithKinFit',
    'HTTv2_lead_mass',
    'HTTv2_lead_pt',
    'HadTop_eta',
    'HadTop_pt',
    'HadTop_pt_CSVsort3rd',
    'HadTop_pt_CSVsort3rd_WithKinFit',
    #'HadTop_pt_boosted',
    #'HadTop_pt_boosted_WithKinFit',
    'HadTop_pt_highestCSV',
    'HadTop_pt_highestCSV_WithKinFit',
    'HadTop_pt_multilep',
    #'HadTop_pt_semi_boosted',
    #'HadTop_pt_semi_boosted_WithKinFit',
    'avg_dr_jet',
    'b1_loose_pt',
    'b1_pt',
    'b2_loose_pt',
    'b2_pt',
    'detabb',
    'detabb_loose',
    'dr_lep1_tau_os',
    'dr_lep2_tau_ss',
    'dr_leps',
    'drbb',
    'drbb_loose',
    #'evtWeight',
    #'fitHTptoHTmass',
    #'fitHTptoHTpt',
    'fittedHadTop_eta',
    'fittedHadTop_pt',
    #'genTopPt',
    #'genTopPt_CSVsort3rd',
    #'genTopPt_CSVsort3rd_WithKinFit',
    #'genTopPt_boosted',
    #'genTopPt_boosted_WithKinFit',
    #'genTopPt_highestCSV',
    #'genTopPt_highestCSV_WithKinFit',
    #'genTopPt_multilep',
    #'genTopPt_semi_boosted',
    #'genTopPt_semi_boosted_WithKinFit',
    #'genWeight',
    'htmiss',
    'lep1_conePt',
    'lep1_eta',
    'lep1_fake_prob',
    'lep1_genLepPt',
    'lep1_pt',
    'lep1_tth_mva',
    'lep2_conePt',
    'lep2_eta',
    'lep2_fake_prob',
    'lep2_genLepPt',
    'lep2_pt',
    'lep2_tth_mva',
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
    'min_lep_eta',
    'mindr_lep1_jet',
    'mindr_lep2_jet',
    'mindr_tau_jet',
    'ptbb',
    'ptbb_loose',
    'ptmiss',
    'tau_eta',
    'tau_fake_prob',
    'tau_genTauPt',
    'tau_mva',
    'tau_pt',
    'N_jetAK12',
    'N_jetAK8',
    #'bWj1Wj2_isGenMatchedWithKinFit',
    #'bWj1Wj2_isGenMatched_CSVsort3rd',
    #'bWj1Wj2_isGenMatched_CSVsort3rd_WithKinFit',
    #'bWj1Wj2_isGenMatched_IHEP',
    #'bWj1Wj2_isGenMatched_boosted',
    #'bWj1Wj2_isGenMatched_boosted_WithKinFit',
    #'bWj1Wj2_isGenMatched_highestCSV',
    #'bWj1Wj2_isGenMatched_highestCSV_WithKinFit',
    #'bWj1Wj2_isGenMatched_semi_boosted',
    #'bWj1Wj2_isGenMatched_semi_boosted_WithKinFit',
    'hadtruth',
    'hadtruth_boosted',
    'hadtruth_semi_boosted',
    'lep1_charge',
    'lep1_isTight',
    'lep1_tau_charge',
    'lep2_charge',
    'lep2_isTight',
    'nBJetLoose',
    'nBJetMedium',
    'nElectron',
    'nHTTv2',
    'nJet',
    #'nLep',
    #'nMuon',
    #'nTau',
    'tau_charge',
    #'tau_isTight',
    #'run',
    'luminosityBlock',
    'event'
]



###============================path to root files

forBDTtraining="forBDTtraining"
channel='2los_1tau'
rootfilepath = path+"*/*.root"
list = glob.glob(rootfilepath)
#print list


#========convert list of filenames to list of tfile objects

root_files=[]
for x in range(0,len(list)):
    try: root_files.append(TFile.Open(list[x]))
    except: continue
#root_files


#=================A dictionary to contain all data
##the list is: ['signal',TTW','TTZ','TT']
if bkg == "TTV":
    myDict = {
    'signal':[],
    'TTW':[],
    'TTZ':[],
    #'TT':[]
    }
elif bkg == "TT":
    myDict = {
    'signal':[],
    #'TTW':[],
    #'TTZ':[],
    'TT':[]
    }
elif bkg == "all":
    myDict = {
    'signal':[],
    'TTW':[],
    'TTZ':[],
    'TT':[]
    }


#====================Filling the dictionary
nEvtTot = 0
for categories in myDict.keys():
    try:
        print "\n\nEntering processes "+categories+"\nRoot files found in category: "
        for root_file in root_files:
            path_in = channel+"_"+forBDTtraining+"/sel/evtntuple/"+categories+"/evtTree"
            evt_tree=root_file.Get(path_in)
            if (evt_tree is not None) :
                if isinstance(evt_tree,ROOT.TTree):
                    print categories+"\t"+root_file.GetName()+' '+str(evt_tree.GetEntries())+' entries.'
                    myDict[categories].append(evt_tree)
                    nEvtTot += evt_tree.GetEntries()
            else:
                continue
    except:
        continue



#======================Setting up TMVA and Loading Data

dataloader = TMVA.DataLoader('dataset')
for branch in myDict['signal'][0].GetListOfBranches():
    for selection in varlist:
        if (selection == branch.GetName()):
            try:
                dataloader.AddVariable(selection)
            except:
                print "\n\n\n\n\n\n\n"+selection+" has problems."

TMVA.Tools.Instance()
TMVA.PyMethodBase.PyInitialize()

output = TFile.Open('TMVA.root', 'RECREATE')
#dataloader.AddSignalTree(myDict['signal'][0], 1.0)


for categories in myDict.keys():
    if categories == 'signal': 
        for i in range(0,(len(myDict[categories]))):
            if myDict[categories][i].GetEntries()!=0:
                dataloader.AddSignalTree(myDict[categories][i], 1.0)
    else:
        for i in range(0,(len(myDict[categories]))):
            if myDict[categories][i].GetEntries()!=0:
                dataloader.AddBackgroundTree(myDict[categories][i], 1.0)
#train_sig = myDict["signal"][0].GetEntries()*frac
#print train_sig
#train_bkg = (nEvtTot - myDict["signal"][0].GetEntries())*frac
#print train_bkg
dataloader.PrepareTrainingAndTestTree(TCut(''),
                                      'nTrain_Signal=22000:nTrain_Background=15000:SplitMode=Random:NormMode=NumEvents:!V')



# ==================================Book methods
factory = TMVA.Factory('TMVAClassification', output,
                       '!V:!Silent:Color:DrawProgressBar:AnalysisType=Classification')


####=============PyKeras 

#====================building the model
model = Sequential()
num_layers=10
num_nodes=25
model.add(Dense(num_nodes, activation='tanh', input_dim=len(varlist)))
for i in range(0,num_layers):
    model.add(Dense(num_nodes, activation='tanh'))
model.add(Dense(2, activation='softmax'))
# ====================Set loss and optimizer
model.compile(loss='mean_squared_error',
              optimizer=SGD(),
              metrics=['binary_accuracy', ])
# =====================Store model to file
model.save('model.h5')
model.summary()

factory.BookMethod(dataloader, TMVA.Types.kPyKeras, 'PyKeras','!H:!V:FilenameModel=model.h5:NumEpochs=5')

factory.BookMethod(dataloader, TMVA.Types.kBDT, 'BDTG','!H:!V:NTrees=1000::BoostType=Grad:Shrinkage=0.30:UseBaggedGrad:GradBaggingFraction=0.6:SeparationType=GiniIndex:nCuts=20:PruneMethod=CostComplexity:PruneStrength=50:NNodesMax=5')


factory.TrainAllMethods()
factory.TestAllMethods()
factory.EvaluateAllMethods()