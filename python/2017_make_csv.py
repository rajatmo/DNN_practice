#!bin/python2

import glob
import os
import ROOT
from ROOT import TFile,TTree,TBranch
import numpy as np
from root_numpy import root2array



rootfilelocationdictionary = {}
csvfilelocationdictionary = {}
categories = ['signal','TT','TTW','TTZ']

for filename in glob.glob('../root/*/*/*/*.root'):
    rootfilenamelists = filename.split("/")
    rootfilenamelists[1] = 'csv'
    csvfilelocationdictionary[rootfilenamelists[-1].split(".")[0]] = "/".join(rootfilenamelists[0:-1])
    rootfilenamelists[1] = 'root'
    rootfilelocationdictionary[rootfilenamelists[-1]] = "/".join(rootfilenamelists[0:-1])

for filenames in csvfilelocationdictionary:
    if not os.path.exists(csvfilelocationdictionary[filenames]):
        os.makedirs(csvfilelocationdictionary[filenames])

for rootfile in rootfilelocationdictionary:
    print 'trying to open '+rootfile
    try: root_tfile = TFile.Open(rootfilelocationdictionary[rootfile]+'/'+rootfile)
    except: print 'cannot open '+rootfile
    for items in categories:
        root_ttree = root_tfile.Get('2los_1tau_forBDTtraining/sel/evtntuple/'+items+'/evtTree')
        if (root_ttree is not None) :
            if isinstance(root_ttree,ROOT.TTree):
                print rootfile+" is of category "+items
                try: chunk_arr=root2array(rootfilelocationdictionary[rootfile]+'/'+rootfile,'2los_1tau_forBDTtraining/sel/evtntuple/'+items+'/evtTree')
                except: continue
                csvfile = open(csvfilelocationdictionary[rootfile.split('.')[0]]+'/'+rootfile.split('.')[0]+'.csv','w')
                for varnames in chunk_arr.dtype.names:
                    csvfile.write(varnames+",")
                csvfile.write('\n')
                for i in range (0,chunk_arr.shape[0]):
                    for varnames in chunk_arr.dtype.names:
                        csvfile.write(str(chunk_arr[varnames][i])+',')
                    csvfile.write( '\n')
                csvfile.close()

