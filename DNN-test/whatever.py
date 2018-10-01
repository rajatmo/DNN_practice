import glob
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
from ROOT import TMVA, TFile, TTree, TCut
from subprocess import call
from os.path import isfile













forBDTtraining="forBDTtraining"
channel='2los_1tau_Tight'
rootfilepath = path+"*/*.root"
list = glob.glob(rootfilepath)
root_files=[]
for x in range(0,len(list)):
	try:	root_files.append(TFile.Open(list[x]))
	except: continue
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

