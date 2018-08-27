#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Aug 23 23:55:46 2018

@author: rajat
"""

import glob
import ROOT


def getTrainVars(channel, trainvar, bdtType, all):
    if channel=="2los_1tau" and all==True  :
        return [
        'HadTop_eta', 'HadTop_pt',
        'avg_dr_jet', 'dr_lep1_tau_os', 'dr_lep2_tau_ss',
        'dr_lepOS_HTfitted', 'dr_lepOS_HTunfitted', 'dr_lepSS_HTfitted', 'dr_lepSS_HTunfitted',
        'dr_leps', 'dr_tau_HTfitted', 'dr_tau_HTunfitted', #'evtWeight',
        'fitHTptoHTmass', 'fitHTptoHTpt', #'genTopPt', 'genWeight',
        'htmiss', 'lep1_conePt', 'lep1_eta', #'lep1_fake_prob', 'lep1_genLepPt',
        'lep1_pt', 'lep1_tth_mva', 'lep2_conePt', 'lep2_eta', #'lep2_fake_prob', 'lep2_genLepPt',
        'lep2_pt', 'lep2_tth_mva', #'lumiScale',
        'mT_lep1', 'mT_lep2', 'mTauTauVis', #'mass_lepOS_HTfitted', 'mass_lepSS_HTfitted',
        'max_lep_eta', 'mbb', 'mbb_loose',  'b1_loose_pt', 'b2_loose_pt', 'drbb_loose', 'detabb_loose',
        'min_lep_eta', 'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
        #'mvaDiscr_2lss', 'mvaOutput_2lss_ttV', 'mvaOutput_2lss_ttbar', 'mvaOutput_hadTopTagger',
        'mvaOutput_hadTopTaggerWithKinFit', 'ptbb', 'ptbb_loose', 'ptmiss', 'tau_eta',
        #'tau_fake_prob', 'tau_genTauPt', 'tau_mva', 'lep2_charge', 'lep2_isTight',
        'tau_pt', 'unfittedHadTop_eta', 'unfittedHadTop_pt', #'bWj1Wj2_isGenMatched',
        'bWj1Wj2_isGenMatchedWithKinFit', #'hadtruth', 'lep1_charge', 'lep1_isTight',
        'lep1_tau_charge', 'nBJetLoose', 'nBJetMedium',
        'nJet', 'nLep', 'nTau', #'tau_charge', 'tau_isTight', 'run', 'lumi', 'evt'
        		]

    if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
        return [
        'avg_dr_jet', 'dr_lep1_tau_os',
        'dr_lep2_tau_ss',
        'dr_leps',
        'lep1_conePt',
        'lep2_conePt',
        'mT_lep1',
        'mT_lep2',
        'mTauTauVis',
        'tau_pt', 'tau_eta',
        'max_lep_eta', 'min_lep_eta', 'lep2_eta','lep1_eta',
        'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
        'mbb', 'ptbb',  #(medium b)
        'mbb_loose', 'ptbb_loose',
        #'b1_loose_pt', 'b2_loose_pt',
        #'drbb_loose', 'detabb_loose',
        'ptmiss', 'htmiss',
        'nBJetLoose',
        'nBJetMedium',
        'nJet',
        ]

    if trainvar=="noHTT" and channel=="2los_1tau" and bdtType=="evtLevelTTV_TTH" and all==False :
        return [
        'avg_dr_jet', #'dr_lep1_tau_os',
        'dr_lep2_tau_ss',
        'dr_leps',
        #'lep1_conePt',
        #'lep2_conePt',
        #'mT_lep1',
        'mT_lep2',
        'mTauTauVis',
        'tau_pt', 'tau_eta',
        #'max_lep_eta', #'min_lep_eta', #'lep2_eta','lep1_eta',
        'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
        #'mbb', 'ptbb', # (medium b)
        'mbb_loose', #'ptbb_loose',
        'ptmiss', #'htmiss',
        #'nBJetLoose',
        #'nBJetMedium',
        'nJet',
        ]

    if trainvar=="HTT" and channel=="2los_1tau" and bdtType=="evtLevelTT_TTH" and all==False :
        return [
        'avg_dr_jet', #'dr_lep1_tau_os',
        'dr_lep2_tau_ss',
        'dr_leps',
        #'lep1_conePt',
        #'lep2_conePt',
        #'mT_lep1',
        'mT_lep2',
        'mTauTauVis',
        'tau_pt', 'tau_eta',
        #'max_lep_eta', 'min_lep_eta', 'lep2_eta','lep1_eta',
        'mindr_lep1_jet', 'mindr_lep2_jet', 'mindr_tau_jet',
        #'mbb', 'ptbb', # (medium b)
        'mbb_loose', #'ptbb_loose',
        'ptmiss', #'htmiss',
        #'nBJetLoose',
        #'nBJetMedium',
        'nJet',
        'mvaOutput_hadTopTaggerWithKinFit',
        'unfittedHadTop_pt',
        #'dr_lepOS_HTfitted', 'dr_lepOS_HTunfitted',
        #'dr_lepSS_HTfitted', 'dr_lepSS_HTunfitted',
        'dr_tau_HTfitted', #'dr_tau_HTunfitted',
        'fitHTptoHTmass', #'fitHTptoHTpt',
        #'HadTop_eta', 'HadTop_pt',
        #'mass_lepOS_HTfitted', 'mass_lepSS_HTfitted',
        ]
        


def load_data_2017(inputPath,channelInTree,variables,criteria,bdtType) :
    print (variables)
    my_cols_list=variables+['key','target',"totalWeight"]
    data = pandas.DataFrame(columns=my_cols_list)
    if bdtType=="evtLevelTT_TTH" : keys=['ttHToNonbb','TTTo2L2Nu','TTTo2L2Nu_PSweights','TTToHadronic','TTToHadronic_PSweights','TTToSemiLeptonic','TTToSemiLeptonic_PSweights']
    if bdtType=="evtLevelTTV_TTH" : keys=['ttHToNonbb','TTWJets','TTZJets']
    if "evtLevelSUM_TTH" in bdtType : keys=['ttHToNonbb','TTWJets','TTZJets','TTTo2L2Nu','TTTo2L2Nu_PSweights','TTToHadronic','TTToHadronic_PSweights','TTToSemiLeptonic','TTToSemiLeptonic_PSweights']
    if bdtType=="all" : keys=['ttHToNonbb','TTWJets','TTZJets','TTTo2L2Nu','TTTo2L2Nu_PSweights','TTToHadronic','TTToHadronic_PSweights','TTToSemiLeptonic','TTToSemiLeptonic_PSweights']
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
            try: tfile = ROOT.TFile(list[ii])
            except : continue
            try: tree = tfile.Get(inputTree)
            except : continue
            if tree is not None :
                try: chunk_arr = tree2array(tree) #,  start=start, stop = stop)
                except : continue
                else :
                    chunk_df = pandas.DataFrame(chunk_arr)
                    #print (len(chunk_df))
                    #print (chunk_df.columns.tolist())
                    chunk_df['proces']=sampleName
                    chunk_df['key']=folderName
                    chunk_df['target']=target
                    chunk_df["totalWeight"] = chunk_df["evtWeight"]
                    if channel=="0l_2tau" :
                        chunk_df["tau1_eta"]=abs(chunk_df["tau1_eta"])
                        chunk_df["tau2_eta"]=abs(chunk_df["tau2_eta"])
                        chunk_df["HadTop1_eta"]=abs(chunk_df["HadTop1_eta"])
                        chunk_df["HadTop2_eta"]=abs(chunk_df["HadTop2_eta"])
                    data=data.append(chunk_df, ignore_index=True)
            else : print ("file "+list[ii]+"was empty")
            tfile.Close()
        if len(data) == 0 : continue
        nS = len(data.ix[(data.target.values == 1) & (data.key.values==folderName) ])
        nB = len(data.ix[(data.target.values == 0) & (data.key.values==folderName) ])
        print (folderName,"length of sig, bkg: ", nS, nB , data.ix[ (data.key.values==folderName)]["totalWeight"].sum(), data.ix[(data.key.values==folderName)]["totalWeight"].sum())
        nNW = len(data.ix[(data.evtWeight.values < 0) & (data.key.values==folderName) ]) 
        print (folderName, "events with -ve weights", nNW )
    print (data.columns.values.tolist())
    n = len(data)
    nS = len(data.ix[data.target.values == 1])
    nB = len(data.ix[data.target.values == 0])
    print (channelInTree," length of sig, bkg: ", nS, nB)
    return data
