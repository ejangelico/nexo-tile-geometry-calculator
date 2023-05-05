import sys
sys.path.append("../classes/")

import LookupTable
import glob
import pickle

if(len(sys.argv) != 3):
    print("Please use: python chain_data.py <path to data> <output filename>")
    sys.exit()

data_path = sys.argv[1]
infiles = glob.glob(data_path+"*.p")
outfile = data_path+sys.argv[2]

chained_tab = None
for f in infiles:
    if("chained" in f): continue #don't load the output file if it was created earlier
    if(chained_tab == None):
        chained_tab = pickle.load(open(f, "rb"))[0]
        continue
    temp_tab = pickle.load(open(f, "rb"))[0]
    chained_tab = chained_tab + temp_tab

pickle.dump([chained_tab], open(outfile, "wb"))
        
