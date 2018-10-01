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
import glob



processes_list=['signal','ttH_hbb','TTW','TTZ','Rares','EWK']


rootfilepath = '../root/forBDTtraining/*/'
rootfilepathnames = glob.glob(rootfilepath)
for folderlocation in rootfilepathnames:
	rootfilelist =rootfilepath+'*.root'
	rootfilelistnames = glob.glob(rootfilelist)
	for rootfileames in rootfilelistnames:
		csvnames=rootfileames.split('.')
		try: 
			print 'creating folder '+'..'+csvnames[2]+'_csv'
			os.mkdir('..'+csvnames[2]+'_csv')
		except:	
			print '..'+csvnames[2]+'_csv'+' already exists'
			pass
		for processes in processes_list:
			path_in ="2los_1tau_forBDTtraining/sel/evtntuple/"+processes+"/evtTree"
			print 'trying to open '+rootfileames
			tmp0=TFile.Open(rootfileames)
			tmp1=tmp0.Get(path_in)
			if (tmp1 is not None) :
				if isinstance(tmp1,ROOT.TTree):
					chunk_arr=root2array(rootfileames,path_in)
					for k in tmp1.GetListOfBranches():
						f=open('..'+csvnames[2]+'_csv/'+k.GetName()+'.csv','w')
						for i in range(0,chunk_arr.shape[0]-1):
							f.write(str(chunk_arr[i][k.GetName()])+'\n')
						f.close()
				

