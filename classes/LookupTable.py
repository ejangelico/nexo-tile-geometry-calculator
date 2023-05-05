import numpy as np 
from shapely.ops import unary_union
import shapely
from shapely.geometry import Polygon
from shapely.geometry.point import Point
from shapely import affinity
import matplotlib.pyplot as plt 
import pandas as pd 
import time
from matplotlib.ticker import FormatStrFormatter
from tqdm.notebook import tqdm
import scipy
import scipy.ndimage as ndimage
import scipy.signal
from scipy.integrate import dblquad 

from Physics import calc_trans_diffusion, calc_long_diffusion, calc_drift_velocity, sample_initial_charge_radius
import TileArray
import Tile

plt.style.use('~/evanstyle.mplstyle')

class LookupTable:
	def __init__(self, tilearray):
		self.tilearray = tilearray #a tile array object from TileArray.py

		self.diff = None #units mm^2/us transverse diffusion, filled in setup_discretatization
		self.vel = None #drift velocity mm/us
		self.coordinates = [[], [], []] #x, y, z coordinates unmeshed
		self.X = None #meshgrid X array 
		self.Y = None #meshgrid Y array
		self.Z = None
		self.coords_vstack = [] #yet another coordinate format, [[x0, y0, z0], [x1, y1, z1], ...]
		self.nx = None
		self.ny = None
		self.nz = None #discretization of coordinate ranges. 
		self.unit_cell = False #whether the map has been trimmed to the unit cell changes how one plots sometimes. 

		self.output_df = pd.DataFrame()


	#adds self to table a; checks to make sure
	#that the x-y coordinates are the same, and preserves
	#a lot of the qualities of the self object, such as the
	#tile object and rim and such. Basically assuming that we
	#are adding only additional z-layers of the dataframe and coordinates. 
	def __add__(self, a):
		#form new table and keep all of the unchanging parameters
		newtab = LookupTable(self.tilearray) 
		newtab.diff = self.diff 
		newtab.vel = self.vel

		errorcheck = []
		#check that x-y coordinates are the same. 
		if(list(self.coordinates[0]) != list(a.coordinates[0])):
			print("X coordinates are not identical in the sum of two tables")
			errorcheck.append(1)
		#check that x-y coordinates are the same. 
		if(list(self.coordinates[1]) != list(a.coordinates[1])):
			print("X coordinates are not identical in the sum of two tables")
			errorcheck.append(2)

		#check that we are added two tables that are either (1) both
		#have not been simulated or (2) both have been simulated, but not
		#a mixture of simulated and unsimulated. This is done by checking
		#that energy_loss has been generated in the output_df for the z coordinates
		#from each table. 

		#first, if the tables are empty, then it is clearly not been simulated
		selflen = len(self.output_df.index)
		alen = len(a.output_df.index)
		if((alen == 0 and selflen != 0) or (alen != 0 and selflen == 0)):
			print("One of the two table summands has been simulated and the other has not")
			errorcheck.append(3)

		#if any errors exist, return self. 
		if(len(errorcheck) != 0):
			return self  

		#now, process the two cases: unsimulated and simulated
		if(alen == 0):
			#xs and ys are the same
			xs = self.coordinates[0]
			ys = self.coordinates[1]
			#append z's, the set_coordaintes routing sorts them correctly
			zs = list(self.coordinates[2]) + list(a.coordinates[2])
			newtab.set_coordinates([xs, ys, zs])
			return newtab
		#otherwise, they have simulated and filled output_df objects
		else:
			#xs and ys are the same
			xs = self.coordinates[0]
			ys = self.coordinates[1]
			#append z's, the set_coordaintes routing sorts them correctly
			zs = list(self.coordinates[2]) + list(a.coordinates[2])
			newtab.set_coordinates([xs, ys, zs])
			newdf = pd.concat([self.output_df, a.output_df], ignore_index=True)
			newtab.set_df(newdf)
			return newtab


	#sets the self.coordinates list of lists, then 
	#regenerates the mshgrid and coords_vstack and ni objects
	def set_coordinates(self, coords):
		self.coordinates = coords 

		#always order z array from most positive to most negative. 
		self.coordinates[2] = sorted(self.coordinates[2], reverse=True)
		xs = self.coordinates[0]
		ys = self.coordinates[1]
		zs = self.coordinates[2] #shorthand
		self.nx = len(xs)
		self.ny = len(ys)
		self.nz = len(zs)
		#meshgrid in case it becomes useful.
		self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs)

		#yet another coordinate format, [[x0, y0, z0], [x1, y1, z1], ...]
		self.coords_vstack = np.vstack(([self.X, self.Y, self.Z])).reshape(3,-1).T

	def get_coordinates(self):
		return self.coordinates

	def get_pixel_pitch(self):
		return np.abs(self.coordinates[0][0] - self.coordinates[0][1])

	#returns dataframe with just relevant z slice
	def get_zslice(self, zindex):
		zval = self.coordinates[2][zindex]
		zmask = (self.output_df['z'] == zval)
		df = self.output_df[zmask]
		#select only relevant columns
		df = df.filter(['x','y','lost_charge', 'diel_charge', 'cond_charge', 'rad_f'], axis=1)
		return df

	def get_n_zslices(self):
		return len(self.coordinates[2])

	def get_xslice(self, xindex=None, xval=None):
		if(xindex is None and xval is None):
			print("Cant get xsclice")
			return
		if(xindex is None):
			xmask = (self.output_df['x'] == xval)
		else:
			xval = self.coordinates[0][xindex]
			xmask = (self.output_df['x'] == xval)

		
		df = self.output_df[xmask]
		#select only relevant columns
		df = df.filter(['z','y','lost_energy', 'rad_f'], axis=1)
		return df


	def set_df(self, df):
		self.output_df = df

	#this function interprets the tile object and then
	#trims to a "unit cell" in the x-y space, removing the
	#edges of the simulation such that one is left with a
	#piece that allows you to construct the entire tile - i.e.
	#separating out the translational symmetry of the tile. This 
	#means that the center point of the unit cell is at a bridge crossing
	#and the edges are 0.5 pitch away from the center. *Note that
	#this ignores the difference between x-over-y and y-over-x crossings
	def trim_to_unit_cell(self, tile_id):
		#the centerpoint of the unit cell is at a different
		#point based on if N is even or odd. 
		t_ex = self.tilearray.tiles[tile_id]["t"]
		t_ex_pos = self.tilearray.tiles[tile_id]["pos"] 
		p = t_ex.get_p()
		N = t_ex.get_N()

		#odd: at N/2*pitch
		if(N % 2 == 1):
			center = 0.5*N*p 
		#even: the crossing is slightly
		#off center relative to the edges. 
		#becomes 0.5*N*p - 0.5*p
		else:
			center = 0.5*N*p - 0.5*p

		#use masks to remove elements of the dataframe
		#outside of the boundary, which is center +- 0.5*p in both x and y
		mask = (self.output_df['x'] <= (center + t_ex_pos[0] + 0.5*p)) & \
				(self.output_df['x'] >= (center + t_ex_pos[0] - 0.5*p)) & \
				(self.output_df['y'] <= (center + t_ex_pos[1] + 0.5*p)) & \
				(self.output_df['y'] >= (center + t_ex_pos[1] - 0.5*p))

		newdf = self.output_df[mask]
		#zero the new dataframes x-y coordinates
		#such that center is at 0,0
		newx = newdf['x'] - center - t_ex_pos[0]
		newy = newdf['y'] - center - t_ex_pos[1]
		newdf['x'] = newx
		newdf['y'] = newy

		#for hit strips, used in analysis later, the
		#strip positions need adjusting as well. 



		self.set_df(newdf)

		#update the coordinates
		xs = sorted(list(set(newdf['x'])))
		ys = sorted(list(set(newdf['y'])))
		zs = sorted(list(set(newdf['z'])), reverse=True)
		self.set_coordinates([xs, ys, zs])

		self.unit_cell = True


	#forms objects that hold info about the
	#discretization of the z axis and transverse space
	#zrange: the range of z coordinates over which to divide by nz
	#nx and ny discretize based on boundaries of the tile object
	
	#search type lets you switch between targetting various regions of a tile array
	def setup_discretization(self, nx, ny, dz, zrange, search_type='full'):
		
		self.nx = nx
		self.ny = ny 

		#example: zrange = [-1, -100], dz = -2 (note negative signs)
		zs = np.arange(max(zrange), min(zrange), dz)
		self.nz = len(zs)


		#get the x-y extent of the coordinates to simulate charge showers in
		x_minmax = None
		y_minmax = None
		if(search_type == 'full'):
			x_minmax, y_minmax = self.tilearray.get_array_xy_extent()

		elif(search_type == 'bulk'):
			#get just a region that is in the bulk of one of the tiles
			#two unit cells wide
			tile_id = 0 #the 0th tile, by default corner is at global coords 0,0
			t_ex = self.tilearray.tiles[tile_id]["t"]
			p = t_ex.get_p()
			N = t_ex.get_N()
			#odd: at N/2*pitch
			if(N % 2 == 1):
				center = 0.5*N*p 
			#even: the crossing is slightly
			#off center relative to the edges. 
			#becomes 0.5*N*p - 0.5*p
			else:
				center = 0.5*N*p - 0.5*p

			#xy extent is one full pitch in either direction
			x_minmax = y_minmax = [center - p, center + p]
		
		elif(search_type == 'gap_corner'):
			#get just the region by a corner of four tiles
			tile_id = 0 #the 0th tile, by default corner is at global coords 0,0
			if(self.tilearray.Ny < 2 or self.tilearray.Nx < 2):
				print("Cant use discretization type 'gap_corner' because the tile array doesn't 2 or more rows AND columns")
				print("Setting to full instead :(")
				x_minmax, y_minmax = self.tilearray.get_array_xy_extent()
			else:
				#go to the first corner. 
				t_ex = self.tilearray.tiles[tile_id]["t"]
				center = t_ex.tile_length + self.tilearray.tgap/2.0
				x_minmax = y_minmax = [center - t_ex.p, center + t_ex.p]



		xs = np.linspace(min(x_minmax), max(x_minmax), nx)
		ys = np.linspace(min(y_minmax), max(y_minmax), ny)

		#saving coordinates for use later in populated pandas dataframe.
		self.coordinates = [xs, ys, zs]

		#meshgrid in case it becomes useful.
		self.X, self.Y, self.Z = np.meshgrid(xs, ys, zs)

		#yet another coordinate format, [[x0, y0, z0], [x1, y1, z1], ...]
		self.coords_vstack = np.vstack(([self.X, self.Y, self.Z])).reshape(3,-1).T

	#efield in V/cm
	#This function does the big computation. It generates a charge
	#cloud at each position setup in discretization_setup and finds
	#which objects in the tilearray overlap with it. The gaussian flag
	#will either (False) use a constant density circular charge cloud with
	#radius equal to the 3-sigma radius of the 0vbb clouds or (True) split
	#the charge cloud into "N_ann" circular annuli weighted by the gaussian 
	#function at the radius of the annuli. N_ann is the number of bins effectively in
	#the 2D gaussian. Then each annulus is checked for overlap. Gaussian is cut off at 4 sigma
	def generate_lookup_table(self, efield, gaussian=False, N_ann=4):
		self.diff = calc_trans_diffusion(efield) #mm^2/us
		self.vel = calc_drift_velocity(efield) #mm/us

		#get initial charge radius based on a simulation set from
		#nexo-offline of 0vbb's. Samples a distribution with mean value of
		#0.88 mm diameter / 2 = 0.44 mm radius, but with a range that extends to 3mm/2. 
		#imported from Physics.py. For gaussian density charge cloud we'll call this 3-sigma radius
		radius_init = 0.44 #sample_initial_charge_radius() #mm radius of charge cloud
		def biv_gaus(y, x, sig):
			norm = (1.0/(2*np.pi*sig*sig))
			return norm*np.exp(-(x**2 + y**2)/(2*sig*sig))
		

		#get fractional area of overlap with all strips, and just discard any that 
		#have less than a threshold value of overlap with circle. 
		thresh = 1e-6 #fraction of area.
		ncoords = len(self.coords_vstack)
		counter = 0
		t0 = time.time() #timing performance
		for z in self.coordinates[2]:
			#calculate the size of the resulting charge cloud
			#assuming circular with square area size_init. drift
			#time is z/self.vel. Then radius increases by sqrt(diff*t)
			#positive z is above the anode.
			if(z > 0): continue
			radius_final = 3*np.sqrt(2*(-1*z/self.vel)*self.diff) + radius_init #factor of 3 is for 
			#create a shapely object representing the charge cloud. Either uniform density circle
			#or a binned gaussian based on the argument to this function. 
			if(gaussian == False):
				#shapely circle, at default resolution 32 there is 2% error on area, at 200 it is 1e-5
				cloud = [(Point(0, 0)).buffer(radius_final, resolution=200)]
				total_charge = [1]
			else:
				#create innermost circle first, then append annuli. 
				#the "total charge" of each annulus is the integral
				#of the bivariate normal distribution from r0 to r1. 
				sigma = radius_final/3
				#go to radii out to 4 sigma
				radii = np.linspace(0, 4*sigma, N_ann+1)

				for i in range(1, len(radii)):
					if(i == 1):
						#circle 
						cloud = [(Point(0, 0)).buffer(radii[0], resolution=200)]
						total_charge = [dblquad(biv_gaus, -radii[i], radii[i], -radii[i], radii[i], args=[sigma])[0]]
						continue
					r0 = radii[i-1]
					r1 = radii[i]
					total_charge.append(dblquad(biv_gaus, -r1, r1, -r1, r1, args=[sigma])[0] - \
			 							dblquad(biv_gaus, -r0, r0, -r0, r0, args=[sigma])[0])
					#annulus is difference between two circles
					p1 = (Point(0,0)).buffer(r1, resolution=200)
					p0 = (Point(0,0)).buffer(r0, resolution=200)
					cloud.append(p1.difference(p0))

				#the total charge adds up to 0.99987, the 4sigma integral value of bivariate gaussian 


			for x in self.coordinates[0]:
				for y in self.coordinates[1]:
					if(counter % 10 == 0):
						print("On coordinate " + str(counter) + " of " + str(ncoords))
					
					#generate the charge cloud at this position by shifting all polygons in cloud
					cloud_t = [shapely.affinity.translate(_, x, y) for _ in cloud]

					#ask the tile array object to compute if each cloud shape overlaps
					#with any of the tile array polygons, and give back detailed info. 
					hit_objects = []
					for i, shape in enumerate(cloud_t):
						hit_objects_temp = self.tilearray.find_overlap(shape, thresh)
						#add a dict item to each hit object representing the
						#charge fraction from that overlap, given information we have here
						#about the total area and charge within each shape (be it a circle or gaussian annulus)
						a_tot = shape.area 
						c_tot = total_charge[i] #charge within that shape 
						for _ in range(len(hit_objects_temp)):
							hit_objects_temp[_]["c_frac"] = c_tot*hit_objects_temp[_]["area"]/a_tot 
						hit_objects = hit_objects + hit_objects_temp #concatenate to the list

					#calculate three variables, conductor charge, dielectric charge, and lost charge
					cond_charge = 0
					diel_charge = 0
					for ho in hit_objects:
						if(ho["type"] == "x" or ho["type"] == "y"):
							cond_charge += ho["c_frac"]
						else:
							diel_charge += ho["c_frac"]

					lost_charge = 1 - cond_charge - diel_charge

					ds = pd.Series(dtype='object')
					ds['x'] = x
					ds['y'] = y 
					ds['z'] = z
					ds['rad_f'] = radius_final
					ds['rad_i'] = radius_init
					ds['cond_charge'] = cond_charge
					ds['diel_charge'] = diel_charge
					ds['lost_charge'] = lost_charge
					ds['hit_objects'] = hit_objects #the full list of dicts for future use in detailed analyses
					self.output_df = pd.concat([self.output_df, ds])
					counter += 1

		print("Total time " + str((time.time() - t0)))


	#plots lost energy as a function of
	#transverse position, at a particular slice in z (integer index of list)
	def plot_charge_location(self, zindex, loc='diel_charge', ax = None, fig = None, show=True):

		#select only relevant columns
		plot_df = self.get_zslice(zindex)
		#make a nice pandas matrix from this data
		#in preparation to plot


		plot_matrix = plot_df.pivot('x', 'y', loc)
		if(ax is None):
			fig, ax = plt.subplots(figsize=(10, 7))
		heat = ax.imshow(plot_matrix, cmap='viridis', \
			extent=[plot_df['x'].min(), plot_df['x'].max(), plot_df['y'].min(), plot_df['y'].max()],\
			interpolation='nearest')
		self.tilearray.plot_array(ax=ax, show=False)
		scat_x = []
		scat_y = []
		for i in self.coordinates[0]:
			for j in self.coordinates[1]:
				scat_x.append(i)
				scat_y.append(j)
		ax.scatter(scat_x, scat_y)


		cbar = fig.colorbar(heat, ax=ax)
		cbar.set_label('fraction of ' + loc, labelpad=3)

		ax.set_xlabel("x-coordinate (mm)")
		ax.set_ylabel("y-coordinate (mm)")
		rf = round(plot_df['rad_f'].iloc[0], 2)
		zval = self.coordinates[2][zindex]
		ax.set_title("deposition at z = " + str(round(zval, 2)) + " mm with final radius " + str(rf) + " mm")
		
		ax.grid(False)


		#plot an example charge-deposition circle for reference as the final
		#diameter of the charge when it reaches the anode 
		circ = Point(scat_x[0], scat_y[0]).buffer(rf)
		ax.plot(*circ.exterior.xy, 'r')


		if(show):
			plt.show()
		return ax




	def plot_lost_energy_unitcell(self, zindex, ax = None, fig = None, show=True):

		#select only relevant columns
		plot_df = self.get_zslice(zindex)
		#make a nice pandas matrix from this data
		#in preparation to plot
		plot_matrix = plot_df.pivot('x', 'y', 'lost_energy')
		if(ax is None):
			fig, ax = plt.subplots(figsize=(10, 7))
		heat = ax.imshow(plot_matrix, vmin = 0, vmax = 0.10, cmap='viridis', \
			extent=[plot_df['x'].min(), plot_df['x'].max(), plot_df['y'].min(), plot_df['y'].max()],\
			interpolation='nearest')
		self.tile.plot_unit_cell(ax, show=False)
		scat_x = []
		scat_y = []
		for i in self.coordinates[0]:
			for j in self.coordinates[1]:
				scat_x.append(i)
				scat_y.append(j)
		ax.scatter(scat_x, scat_y)

		ax.set_xlim([plot_df['x'].min(), plot_df['x'].max()])
		ax.set_ylim([plot_df['y'].min(), plot_df['y'].max()])


		cbar = fig.colorbar(heat, ax=ax)
		cbar.set_label('fraction of charge lost', labelpad=3)

		ax.set_xlabel("x-coordinate (mm)")
		ax.set_ylabel("y-coordinate (mm)")
		rf = round(plot_df['rad_f'].iloc[0], 2)
		zval = self.coordinates[2][zindex]
		ax.set_title("deposition at z = " + str(round(zval, 2)) + " mm with final radius " + str(rf) + " mm")
		
		ax.grid(False)


		#plot an example charge-deposition circle for reference as the final
		#diameter of the charge when it reaches the anode 
		pit = self.tile.get_p()
		N = self.tile.get_N()
		center = 0
		circ = Point(center, center).buffer(rf)
		ax.plot(*circ.exterior.xy, 'r')


		if(show):
			plt.show()
		return ax

	#just plain histogram all lost energy values
	def hist_lost_energy(self, zindex):
		zval = self.coordinates[2][zindex]
		zmask = (self.output_df['z'] == zval)

		fig, ax = plt.subplots()
		ax.hist(self.output_df['lost_energy'][zmask])
		plt.show()


	#plots in the x-z dimension, loss, similar to
	#PDE photon transport map of the other papers
	def plot_x_z_map(self, outdir=None):
		#getting straight from the dataframe due to trimming of edges
		unique_xcoords = sorted(set(list(self.output_df['x'])))
		for i, x in enumerate(unique_xcoords):
			plot_df = self.get_xslice(xval=x)
			plot_matrix = plot_df.pivot('z', 'y', 'lost_energy')
			fig, ax = plt.subplots(figsize=(17, 7), ncols=2)
			heat = ax[0].imshow(plot_matrix, vmin = 0, vmax = 0.10, cmap='viridis', \
				extent=[plot_df['y'].min(), plot_df['y'].max(), plot_df['z'].min(), plot_df['z'].max()],\
				aspect='auto', origin='lower')

		


			cbar = fig.colorbar(heat, ax=ax[0])
			cbar.set_label('fraction of charge lost', labelpad=3)
			ax[0].set_title("Y-coordinate: " + str(round(x, 2)))
			ax[0].set_xlabel("x-coordinate (mm)")
			ax[0].set_ylabel("z-coordinate (mm)")
			ax[0].grid(False)
			if(self.unit_cell):
				self.tile.plot_unit_cell(ax[1], show=False)
			else:
				self.tile.plot_strips(ax[1], show=False)

			ax[1].set_xlim([self.output_df['x'].min(), self.output_df['x'].max()])
			ax[1].set_ylim([self.output_df['y'].min(), self.output_df['y'].max()])
			ax[1].axhline(x, color='r', linewidth=3)

			
			if(outdir is None):
				plt.show()
			else:
				plt.savefig(outdir+str(i).zfill(6)+".png", bbox_inches='tight')

			plt.clf()
			plt.close()




	#Create two 45 degree diagonal slices, one through the center
	#of the pads and one intersecting the strip-crossings. Create
	#these slices as close to center of the tile as possible (y = x
	#and y = x + 0.5pitch)
	def overlay_sliced_lost_energy(self, zindex, show = True, ax = None, fig = None):

		#if it has been trimmed to unit cell, call another internal function
		if(self.unit_cell):
			return self.overlay_sliced_lost_energy_unitcell(zindex, show, ax, fig)

		pitch = self.tile.get_p()
		pitch45 = pitch/np.sqrt(2) #"pitch" at the 45 degree rotated tile. 
		#parametrized 45 degree rotated lines 
		y1 = lambda x: x #function for slice 1
		y2 = lambda x: x + pitch/2.0 #function for slice 2
		discretization = self.get_pixel_pitch() #unrotated space between x coordinates
		l = np.array(self.coordinates[0]) #x coords act as parameter for line functions
		#debug: show the slices
		"""
		fig, ax = plt.subplots()
		ax.plot(l, y1(l))
		ax.plot(l, y2(l))
		self.plot_lost_energy(zindex, ax, fig, show = False)
		plt.show()
		"""

		#remove unneeded data
		df = self.get_zslice(zindex)
		#meshgrid x and y
		X, Y = np.meshgrid(self.coordinates[0], self.coordinates[1])
		#now following https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
		combined_xy = np.dstack([X.ravel(), Y.ravel()])[0]
		#reshape the list of query points to be [[x, x, x, ...],[y, y, y...]]
		slice1 = []
		slice2 = []
		#slice 2 is in an awkward location where it dithers to find the closest
		#point in the discrete 2D x-y map. I'm going to make a windowed straight line
		#that is not discretized, by saving the first and last x-points within the correct
		#range of the tiles boundaries. 
		plot_slice2_xs = [l[0]]
		plot_slice2_ys = [y2(l[0])]
		for _ in l:
			#keep within bounds of the tile
			if(df['y'].min() <= y1(_) <= df['y'].max()):
				slice1.append([_, y1(_)])
			if(df['y'].min() <= y2(_) <= df['y'].max() \
				and _ > 0):
				slice2.append([_, y2(_)])
			if(y2(_) >= df['y'].max() and len(plot_slice2_ys) == 1):
				plot_slice2_xs.append(_)
				plot_slice2_ys.append(y2(_))

		slice1 = np.array(slice1)
		slice2 = np.array(slice2)

		def do_kdtree(combined_x_y_arrays,points):
		    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
		    dist, indexes = mytree.query(points)
		    return indexes

		indices1 = do_kdtree(combined_xy, slice1)
		indices2 = do_kdtree(combined_xy, slice2)
		
		#debug: show the slices
		"""
		fig, ax = plt.subplots()
		ax.scatter([combined_xy[i][0] for i in indices1], [combined_xy[i][1] for i in indices1])
		ax.scatter([combined_xy[i][0] for i in indices2], [combined_xy[i][1] for i in indices2])
		self.plot_lost_energy(zindex, ax, fig, show = False)
		plt.show()
		"""
		
		

		loss1 = []
		x1 = []
		for i1 in indices1:
			xymask = (df['x'] == combined_xy[i1][0]) & (df['y'] == combined_xy[i1][1])
			loss1.append(list(df[xymask]['lost_energy'])[0])
			x1.append(combined_xy[i1][0])

		loss2 = []
		x2 = []
		for i2 in indices2:
			xymask = (df['x'] == combined_xy[i2][0]) & (df['y'] == combined_xy[i2][1])
			loss2.append(list(df[xymask]['lost_energy'])[0])
			x2.append(combined_xy[i2][0])

		if(ax is None):
			fig, ax = plt.subplots(figsize=(20, 7), ncols=2)
		ax[0].plot(x1, loss1, 'o-', label='slice 1', color='chocolate')
		ax[0].plot(x2, loss2, 'o-', label='slice 2', color='darkorange')
		ax[0].set_ylim([-0.02, 0.2])
		ax[1].plot([combined_xy[i][0] for i in indices1], [combined_xy[i][1] for i in indices1], label='slice 1', linewidth=3, color='chocolate')
		#ax[1].plot([combined_xy[i][0] for i in indices2], [combined_xy[i][1] for i in indices2], label='slice 2', linewidth=3, color='darkorange')
		ax[1].plot(plot_slice2_xs, plot_slice2_ys, label='slice 2', linewidth=3, color='darkorange')
		self.plot_lost_energy(zindex, ax[1], fig, show = False)

		ax[0].legend()
		#ax[1].legend()

		ax[0].set_xlabel("x-position along slice (mm)")
		ax[0].set_ylabel("charge fraction lost")

		if(show):
			plt.show()
		return ax
		

	def overlay_sliced_lost_energy_unitcell(self, zindex, show = True, ax = None, fig = None):
		pitch = self.tile.get_p()
		pitch45 = pitch/np.sqrt(2) #"pitch" at the 45 degree rotated tile. 
		#parametrized 45 degree rotated lines 
		y1 = lambda x: x #function for slice 1
		y2 = lambda x: x + pitch/2.0 #function for slice 2
		discretization = self.get_pixel_pitch() #unrotated space between x coordinates
		l = np.array(self.coordinates[0]) #x coords act as parameter for line functions
		#debug: show the slices
		
		"""
		fig, ax = plt.subplots()
		ax.plot(l, y1(l))
		ax.plot(l, y2(l))
		self.plot_lost_energy(zindex, ax, fig, show = False)
		plt.show()
		"""
		

		#remove unneeded data
		df = self.get_zslice(zindex)
		#meshgrid x and y
		X, Y = np.meshgrid(self.coordinates[0], self.coordinates[1])
		#now following https://stackoverflow.com/questions/10818546/finding-index-of-nearest-point-in-numpy-arrays-of-x-and-y-coordinates
		combined_xy = np.dstack([X.ravel(), Y.ravel()])[0]

		#reshape the list of query points to be [[x, x, x, ...],[y, y, y...]]
		slice1 = []
		slice2 = []
		#slice 2 is in an awkward location where it dithers to find the closest
		#point in the discrete 2D x-y map. I'm going to make a windowed straight line
		#that is not discretized, by saving the first and last x-points within the correct
		#range of the tiles boundaries. 
		plot_slice2_xs = [l[0]]
		plot_slice2_ys = [y2(l[0])]
		for _ in l:
			#keep within bounds of the tile
			if(df['y'].min() <= y1(_) <= df['y'].max()):
				slice1.append([_, y1(_)])
			if(df['y'].min() <= y2(_) <= df['y'].max()):
				slice2.append([_, y2(_)])
			if(y2(_) >= df['y'].max() and len(plot_slice2_ys) == 1):
				plot_slice2_xs.append(_)
				plot_slice2_ys.append(y2(_))

		slice1 = np.array(slice1)
		slice2 = np.array(slice2)

		def do_kdtree(combined_x_y_arrays,points):
		    mytree = scipy.spatial.cKDTree(combined_x_y_arrays)
		    dist, indexes = mytree.query(points)
		    return indexes

		indices1 = do_kdtree(combined_xy, slice1)
		indices2 = do_kdtree(combined_xy, slice2)
		
		#debug: show the slices
		"""
		fig, ax = plt.subplots()
		ax.scatter([combined_xy[i][0] for i in indices1], [combined_xy[i][1] for i in indices1])
		ax.scatter([combined_xy[i][0] for i in indices2], [combined_xy[i][1] for i in indices2])
		self.plot_lost_energy(zindex, ax, fig, show = False)
		plt.show()
		"""
		
		

		loss1 = []
		x1 = []
		for i1 in indices1:
			xymask = (df['x'] == combined_xy[i1][0]) & (df['y'] == combined_xy[i1][1])
			loss1.append(list(df[xymask]['lost_energy'])[0])
			x1.append(combined_xy[i1][0])

		loss2 = []
		x2 = []
		for i2 in indices2:
			xymask = (df['x'] == combined_xy[i2][0]) & (df['y'] == combined_xy[i2][1])
			loss2.append(list(df[xymask]['lost_energy'])[0])
			x2.append(combined_xy[i2][0])

		if(ax is None):
			fig, ax = plt.subplots(figsize=(20, 7), ncols=2)
		ax[0].plot(x1, loss1, 'o-', label='slice 1', color='chocolate')
		ax[0].plot(x2, loss2, 'o-', label='slice 2', color='darkorange')
		ax[0].set_ylim([-0.02, 0.2])
		ax[1].plot([combined_xy[i][0] for i in indices1], [combined_xy[i][1] for i in indices1], label='slice 1', linewidth=3, color='chocolate')
		#ax[1].plot([combined_xy[i][0] for i in indices2], [combined_xy[i][1] for i in indices2], label='slice 2', linewidth=3, color='darkorange')
		ax[1].plot(plot_slice2_xs, plot_slice2_ys, label='slice 2', linewidth=3, color='darkorange')
		self.plot_lost_energy(zindex, ax[1], fig, show = False)

		ax[0].legend()
		#ax[1].legend()

		ax[0].set_xlabel("x-position along slice (mm)")
		ax[0].set_ylabel("charge fraction lost")

		if(show):
			plt.show()
		return ax


	#if the table is representing a unit cell,
	#tile the unit cell into an n x n array of unit cells
	#for easy coordinate manipulations
	def tile_unit_cell(self, n):
		if(self.unit_cell == False):
			print("Will not tile the table, as this is not a unit cell table")
			return

		p = self.tile.get_p() #pitch

		#for odd n, loop symmetric about the cell. 
		#so for n = 3, do -1, 0, 1
		if(n % 2 == 1):
			looper = range(int(-n/2), int(n/2)+1)
		#for even n, have the cell be slightly offset
		#by 1 unit to the left, so n = 4 is -1, 0, 1, 2 
		else:
			looper = range(int(-n/2)+1, int(n/2) + 1)

		#get the x and y coordinates of
		#center positions of the added 
		#unit cells, based on nxn array about
		#the center. 
		full_df = self.output_df  #we will concatenate to this DF
		for i in looper:
			x = p*i #pitch multiplier
			for j in looper:
				y = p*j #pitch multiplier
				if(x == 0 and y == 0):
					continue
				newdf = None
				newdf = self.output_df.copy() #unit cell 
				#shift the unit cell
				newx = newdf['x'] - x
				newy = newdf['y'] - y
				newdf['x'] = newx
				newdf['y'] = newy
				
				#concatenate to the full_df 
				full_df = pd.concat([full_df, newdf], ignore_index=True) 


		#the resulting DF has duplicate rows at the edges
		#of the unit cell. Only consider the coordinate subsets
		full_df = full_df.drop_duplicates(subset=['x', 'y', 'z'], ignore_index=True)

		#finalize the changes and set new df
		self.set_df(full_df)

		#update the coordinates
		xs = sorted(list(set(full_df['x'])))
		ys = sorted(list(set(full_df['y'])))
		zs = sorted(list(set(full_df['z'])), reverse=True)
		self.set_coordinates([xs, ys, zs])




		#given an input series with the same column names
	#as the table dataframe, perform a minimization algorithm
	#that finds the best match position and energy given 
	#experimental priors and not truth info. 
	def find_best_match(self, in_ser):

		#unfortunately there is an issue at the data generation level
		#that needs to be changed next round. When doing computations
		#at the unit-cell level, where the tile is shifted by a constant
		#x-y offset, the hit_x_strips lists are not adjusted. So the coordinates
		#of strips are referencing an old coordinate system. It is difficult
		#to adjust the tables at this stage. Instead, it is done here. 
		#NOTE, the in_ser['hit_x_strips'] has keys that are correctly in
		#the unit cell configuration, whereas the self.output_df is incorrect. 

		#the centerpoint of the unit cell is at a different
		#point based on if N is even or odd. 
		p = self.tile.get_p()
		N = 3

		#odd: at N/2*pitch
		if(N % 2 == 1):
			center = 0.5*N*p 
		#even: the crossing is slightly
		#off center relative to the edges. 
		#becomes 0.5*N*p - 0.5*p
		else:
			center = 0.5*N*p - 0.5*p

		strip_posns = list((self.tile.get_strips()[0]).keys())
		strip_posns = [_ - center for _ in strip_posns]
		nstrips = len(strip_posns)

		#create a 2D image (numpy array) representing the
		#hit-strip amplitudes as a function of strip position. 
		in_img = np.zeros((nstrips, nstrips))
		for pos, amp in in_ser['hit_x_strips'].items():
			idx = strip_posns.index(pos)
			in_img[:,idx] += amp 
		for pos, amp in in_ser['hit_y_strips'].items():
			idx = strip_posns.index(pos)
			in_img[idx,:] += amp 

		#get guess coordinates based on the center of mass
		#of the image of strip amplitudes
		guess_x, guess_y = ndimage.measurements.center_of_mass(in_img)
		guess_x = (guess_x - int(nstrips/2))*2*p
		guess_y = (guess_y - int(nstrips/2))*2*p
		

		#we assume that we know z from timing of SiPMs 
		known_z = in_ser['z']
		#presently, the random generation of z values is such that
		#the z's exactly match the data generated in lookup table. 
		#This is to remove any jitter in the MC due to unknown z values. 
		#Eventually, the lookup tables will be generated with a segmentation
		#that is x1/4 the resolution or something. 
		search_df = self.output_df[self.output_df["z"] == known_z]

		#a mimimization of differences of hit x-strips
		# in x/y coords is formed by brute force
		#scanning every point within a certain range of the guess
		#point. the range can be defined by assuming the guess is 
		#good to within ~X millimeters. 

		x_unc = 6 #mm
		y_unc = 6 #mm 
		mask = (search_df['x'] > (guess_x - x_unc/2)) \
				& (search_df['x'] < (guess_x + x_unc/2)) \
				& (search_df['y'] > (guess_y - y_unc/2)) \
				& (search_df['y'] < (guess_y + y_unc/2))

		search_df = search_df[mask]

		
		#if there are no table entries in the search range,
		#throw a message to the user
		if(len(search_df.index) == 0):
			print("Your lookup table does not have a point generated near the guess, ", end='')
			print("(" + str(guess_x) + ", " + str(guess_y) + ")")
			print("Returning null values. Check your dataframe and generated MC points")
			return None, None

		result_list = []
		for _, row in search_df.iterrows():
			res_ser = {}
			res_ser['x'] = row['x']
			res_ser['y'] = row['y']
			#create a 2D image (numpy array) representing the
			#hit-strip amplitudes as a function of strip position. 
			lookup_img = np.zeros((nstrips, nstrips))
			for pos, amp in row['hit_x_strips'].items():
				idx = strip_posns.index(pos - center)
				lookup_img[:,idx] += amp 
			for pos, amp in row['hit_y_strips'].items():
				idx = strip_posns.index(pos - center)
				lookup_img[idx,:] += amp 

			#compute absolute differences between the images
			sub_img = np.abs(in_img - lookup_img)
			sqerr = np.sum(sub_img)

			res_ser['sqerr'] = sqerr

			#append to full result dataframe
			result_list.append(res_ser)

		best_match = result_list[0]
		for _ in result_list:
			if(_['sqerr'] < best_match['sqerr']):
				best_match = _ 


		#use the lookup table at this position to determine the guess of the 
		#energy that has been lost due to dielectric
		mask = (search_df['x'] == best_match['x']) & (search_df['y'] == best_match['y'])
		table_match = search_df[mask]


		fig, ax = plt.subplots(figsize=(10,8))
		plot_matrix = search_df.pivot('x','y','lost_energy')

		heat = ax.imshow(plot_matrix, cmap='viridis', \
		                 extent=[search_df['x'].min(), search_df['x'].max(), search_df['y'].max(), search_df['y'].min()],\
		                 interpolation='nearest')
		cbar = fig.colorbar(heat, ax=ax)
		cbar.set_label('sum of square strip amplitude differences', labelpad=3)
		ax.scatter([best_match['x']], [best_match['y']], color='red', s=200, label="Reconstructed")
		ax.scatter([in_ser['x']], [in_ser['y']], color='m', s=200, label="True")
		ax.scatter([guess_x], [guess_y], color='g', s=200, label="Initial guess")
		ax.set_xlabel("table x-coordinate (mm)")
		ax.set_ylabel("table y-coordinate (mm)")
		ax.set_title("at a depth of {:.1f} mm with charge radius {:.2f} mm".format(known_z, in_ser['rad_f']))
		ax.legend()
		plt.show()



		return table_match['lost_energy'].iloc[0], np.array([best_match['x'], best_match['y']])






			









		
		






