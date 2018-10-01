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




file=TFile.Open('../root/forBDTtraining/ttHToNonbb_M125_powheg/ttHToNonbb_M125_powheg_forBDTtraining_central_1.root')
print type(file)
tree = file.Get()

