import itertools as it
import numpy as np
from root_numpy import root2array, stretch
from numpy.lib.recfunctions import append_fields
from itertools import product
from ROOT.Math import PtEtaPhiEVector,VectorUtil
import ROOT
from ROOT import TFile,TTree,TBranch
import math , array
import os



exp0=TFile.Open('../root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/ttHJetToNonbb_M125_amcatnlo_forBDTtraining_central_1.root')
exp1=exp0.Get("2los_1tau_Tight/sel/evtntuple/signal/evtTree")
chunk_arr = root2array('../root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/ttHJetToNonbb_M125_amcatnlo_forBDTtraining_central_1.root',"2los_1tau_Tight/sel/evtntuple/signal/evtTree")

foldername='csv/'
for k in exp1.GetListOfBranches():
	print k.GetName()
	f=open('../root/forBDTtraining/ttHJetToNonbb_M125_amcatnlo/'+foldername+k.GetName()+'.csv','w')
	for i in range(0,chunk_arr.shape[0]-1):
		f.write(str(chunk_arr[i][k.GetName()])+'\n')
	f.close()
