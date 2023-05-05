import Tile
import numpy as np 
import matplotlib.pyplot as plt 

plt.style.use("~/evanstyle.mplstyle")

#creates an array of tiles and handles all operations that require
#identification of tiles, individual channels, local/global coordinates,
#etc. 
class TileArray:
    def __init__(self, single_tile, Nrows, Ncols, tile_gap):
        self.N = Nrows*Ncols
        self.Ny = Nrows
        self.Nx = Ncols
        self.t_ex = single_tile #stands for "tile example", the template tile of which to array. 
        self.tgap = tile_gap #space between edges of tiles. 

        #related to keeping track of coordinates and tile identification
        self.tiles = {}
        self.ids = range(self.N)
        for i in self.ids:
            self.tiles[i] = {"t": None, "pos": [None, None]} #"t" is tile object, "pos" is x-y of corner of tile.
        

    def build_array(self):
        #get the coordinate shift needed to create a tile gap of self.tgap
        tile_shift = self.t_ex.tile_length + self.tgap
        id_count = 0
        for i in range(self.Nx):
            for j in range(self.Ny):
                t_temp = Tile.Tile(self.t_ex.p, self.t_ex.g, self.t_ex.N, self.t_ex.tile_rim, self.t_ex.bridge_wid)
                t_temp.build_tile()
                t_temp.shift_all_polygons([i*tile_shift, j*tile_shift])
                self.tiles[id_count]["t"] = t_temp
                self.tiles[id_count]["pos"] = np.array([i*tile_shift, j*tile_shift])
                id_count += 1

    #find the maximum and minimum x and y coordinate to fully
    #cover all polygons in the tile array
    def get_array_xy_extent(self):
        #get the x-y extent of each tile using their outlines
        tile_shift = self.t_ex.tile_length + self.tgap
        yextent = tile_shift*self.Ny
        xextent = tile_shift*self.Nx
        return [0, xextent], [0, yextent]

    def plot_array(self, ax=None, show=True):
        if(ax is None):
            fig, ax = plt.subplots()
        for i in self.ids:
            self.tiles[i]["t"].plot_strips(ax=ax, show=False)


        ax.set_aspect('equal')
        ax.set_xlabel("[mm]")
        ax.set_ylabel("[mm]")
        if(show):
            plt.show()
        return ax

    #for a given shapely shape, find the overlap with 
    #various objects in the tile array
    def find_overlap(self, shape, thresh):
        #bug with shapely creates warnings that will slow down the run. 
        #turn those warnings off
        initial_settings = np.seterr()
        np.seterr(invalid="ignore")
        hit_objects = [] #[{"tile_id": , "type": 'x', 'y', 'diel', "strip_pos": pos in strip array, "area": overlap area}
        for id in self.ids:
            t = self.tiles[id]["t"]
            #first, for efficiency, check if this shape is within the
            #tile outline of this tile. 
            outline = t.tile_outline
            #check if there is an intersection, boolean function
            outline_overlap = outline.intersects(shape)
            if(outline_overlap == False):
                continue #to the next tile
            
            #go through strip polygons and calc overlap
            tot_conductor = 0
            for k, s in t.x_strip_polys.items():
                if(s.intersects(shape)):
                    ov = s.intersection(shape).area 
                    if(ov > thresh):
                        hit_objects.append({"tile_id": id, "type": "x", "strip_pos": k, "area": ov})
                        tot_conductor += ov
            for k, s in t.y_strip_polys.items():
                if(s.intersects(shape)):
                    ov = s.intersection(shape).area 
                    if(ov > thresh):
                        hit_objects.append({"tile_id": id, "type": "y", "strip_pos": k, "area": ov})
                        tot_conductor += ov
        
            #the amount on dielectric is defined as the overlap 
            #on the outline interior minus the total overlap 
            #on the conductors
            hit_objects.append({"tile_id": id, "type":"diel", "strip_pos": None, "area": (outline_overlap - tot_conductor)})
        np.seterr(**initial_settings)
        return hit_objects






