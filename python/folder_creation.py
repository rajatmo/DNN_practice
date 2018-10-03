#!bin/python3

import glob
import os



filelocationdictionary = {}

for filename in glob.iglob('../root/**/*.root', recursive=True):
    rootfilenamelists = filename.split("/")
    rootfilenamelists[1] = 'csv'
    filelocationdictionary[rootfilenamelists[-1]] = "/".join(rootfilenamelists[0:-1])
    print(filelocationdictionary)

for filenames in filelocationdictionary:
    if not os.path.exists(filelocationdictionary[filenames]):
        os.makedirs(filelocationdictionary[filenames])


