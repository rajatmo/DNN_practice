import json
import ROOT

import argparse
import datafile

with open('../json/config.json') as json_file:
    data = json.load(json_file)

parser = argparse.ArgumentParser()


for ite in data['optionParserData']:
    if ('type' in ite):
        if ite['type'] == 'string':
            print (ite['name']," type = string")
            parser.add_argument(ite['name'], dest=ite['var_name'],
                    default=ite['default'], type=str,
                    help=ite['help'])
        elif ite['type'] == 'int':
            print (ite['name']," type = int")
            parser.add_argument(ite['name'], dest=ite['var_name'],
                    default=ite['default'], type=int,
                    help=ite['help'])
        elif ite['type'] == 'float':
            print (ite['name']," type = float")
            parser.add_argument(ite['name'], dest=ite['var_name'],
                    default=ite['default'], type=float,
                    help=ite['help'])
        else: print (ite['name']," type is not supported.")
    elif ('action' in ite):
        if ite['action'] == 'store_true':
           print (ite['name']," action = store true ")
           parser.add_argument(ite['name'], dest=ite['var_name'],
                   default=ite['default'], action = "store_true",
                   help=ite['help'])
        else: print(ite['name'], "type not supported.")
    else:
        print("please specify type/action in json file.")

args = parser.parse_args()

#def load_data_2017(inputPath, channelInTree, variables, criteria, bdtType) :
    

#print(data['file_locations_by_channel'])

for ite in data['file_locations_by_channel']:
    if ('channel' in ite):
        print (ite['channel'])
        data1 = load_data_2017(inputPath=ite['inputPath'], channelInTree=ite['channelInTree'], variables=data['trainVarList'], bdtType = )



data['trainVarList']



"""
def fib (n):
    if (n==1):
        return 1
    elif (n==2):
        return 1
    else:
        return fib(n-1)+fib(n-2)

def Main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
                       "num",
                       help = "Enter n for the nth fibonacci number",
                       type = int
                       )
    args = parser.parse_args()
    result = fib(args.num)
    print ("The "+str(args.num)+"th fib number is "+str(result))
"""