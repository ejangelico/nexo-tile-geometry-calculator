import numpy as np 
import matplotlib.pyplot as plt 
import sys 
import pickle

sys.path.append("../classes/")

import Physics
import LookupTable
import Tile
import TileArray




if __name__ == "__main__":
    if(len(sys.argv) != 7):
        print("python generate_data.py <NX> <NY> <z step> <starting z> <ending z> <output directory and filename no tag>")
        sys.exit()

    argv = [int(_) for _ in sys.argv[1:-1]]
    pitch = 6.0 #mm 
    gap = 0.085 #mm
    bridge_wid = 0.05 #mm
    N = 16
    tile_rim = 0.05 #mm
    tile_gap = 1 #mm

    t = Tile.Tile(pitch, gap, N, tile_rim, bridge_wid)
    tarr = TileArray.TileArray(t, 2, 2, tile_gap)
    tarr.build_array()
    tab = LookupTable.LookupTable(tarr)
    tab.setup_discretization(argv[0], argv[1], argv[2], [argv[3], argv[4]])
    tab.generate_lookup_table(380, gaussian=True, N_ann=5)
    pickle.dump([tab], open(sys.argv[-1]+".p", "wb"))
