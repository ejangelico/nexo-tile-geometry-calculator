import numpy as np 
import sys 
import shapely
from shapely.geometry import Polygon, mapping
from shapely.ops import unary_union
import matplotlib.pyplot as plt 
#import fiona
import json
import subprocess

plt.style.use('~/evanstyle.mplstyle')


class Tile:
	def __init__(self, pitch, gap, N, tile_rim, bridge_wid = None):
		#pad properties
		self.p = pitch #center to center pitch of pads of two parallel strips
		self.g = gap #gap between pads, perpendicular to square side of pad
		self.sl = np.sqrt(2)*(pitch/2.0) - gap 
		self.dsl = pitch - np.sqrt(2)*gap #"diagonal side length"
		self.tile_length = N*pitch + 2*tile_rim

		#tile related properties
		self.N = N #number of strips for each direction. this also sets the square outer tile dimension. 
		self.bridge_wid = bridge_wid #if none, will build no bridges and will just be a pad array
		self.tile_rim = tile_rim #the distance from the edge of the tile to the edge of the nearest electrode, made of dielectric

		self.polygons = [] #list of all, non-unionized polygons
		self.pad_polys = [] #List of all non-unionized pads only polygons
		self.x_strip_polys = {} #x-strips, individually unionized, labeled by position in y axis
		self.y_strip_polys = {} #y-strips, individually unionized, labeled by position in x axis
		self.tile_outline = None #polygon representing the tile outline


	#return shapely polygon objects
	def get_strips(self):
		return self.x_strip_polys, self.y_strip_polys

	def get_p(self):
		return self.p

	def get_N(self):
		return self.N


	#direc is string, 'x' or 'y'
	#position is the position along the
	#coordinate 'x' or 'y'. 
	def build_strip(self, pos, direc):
		#here r is used to represent either x or y
		coords = [] #2-tuples, (x,y) of square pads
		polys = [] #list of polygons, to union at the end. 

		#4 coordinates of each square. order matters
		sq_coords = [np.array([0, 0.5*self.dsl]), \
					np.array([-0.5*self.dsl, 0]), \
					np.array([0, -0.5*self.dsl]), \
					np.array([0.5*self.dsl, 0])]

		#a chopped in half pad, with long edge on the left
		chopped_left = [np.array([0, 0.5*self.dsl]), \
					np.array([0, -0.5*self.dsl]), \
					np.array([0.5*self.dsl, 0])]

		chopped_right = [np.array([0, 0.5*self.dsl]), \
					np.array([-0.5*self.dsl, 0]), \
					np.array([0, -0.5*self.dsl])]

		chopped_top = [np.array([-0.5*self.dsl, 0]), \
					np.array([0, -0.5*self.dsl]), \
					np.array([0.5*self.dsl, 0])]

		chopped_bottom = [np.array([0, 0.5*self.dsl]), \
					np.array([-0.5*self.dsl, 0]), \
					np.array([0.5*self.dsl, 0])]

		cen = None #will be filled with center of each pad square
		#N+1 in the loop is because we will cut the edges
		#so that a "half pad" exists at the endpoints. 
		for n in range(self.N + 1):
			if(direc == 'x'):
				cen = np.array([n*self.p, pos]) #center of square
				if(n == 0):
					#use triangular coordinates, a chopped off pad. 
					coords = [tuple(_ + cen) for _ in chopped_left]
				elif(n == self.N):
					coords = [tuple(_ + cen) for _ in chopped_right]
				else:
					coords = [tuple(_ + cen) for _ in sq_coords] #square points shifted by center
			elif(direc == 'y'):
				cen = np.array([pos, n*self.p])
				if(n == 0):
					#use triangular coordinates, a chopped off pad. 
					coords = [tuple(_ + cen) for _ in chopped_bottom]
				elif(n == self.N):
					coords = [tuple(_ + cen) for _ in chopped_top]
				else:
					coords = [tuple(_ + cen) for _ in sq_coords] #square points shifted by center
			else:
				print("Please input direction either 'x' or 'y'")
				return None

			
			polys.append(Polygon(coords))
		#at this point, the pads themselves are done, so we'll save them
		self.pad_polys += polys


		#Bridges: only build if self.bridge_wid is not none. 
		if(self.bridge_wid is not None):
			#only build if the bridge width is small enough such that
			#it fits in the pointed gap between pads. 
			hl = self.bridge_wid/2.0 #half bridge width
			if(hl < np.sqrt(2)*self.g/2.0):
				if(direc == 'x'):
					coords = [[0, pos+hl], [0, pos-hl], [self.N*self.p, pos-hl], [self.N*self.p, pos+hl]]
					polys.append(Polygon(coords))
				if(direc == 'y'):
					coords = [[pos+hl, 0], [pos-hl, 0], [pos-hl, self.N*self.p], [pos+hl, self.N*self.p]]
					polys.append(Polygon(coords))
			else:
				print("Bridge width is too large! Maximum (determined by pitch and gap) is " + str(np.sqrt(2)*self.g))


		#save all polygons just for safe keeping
		self.polygons += polys

		#unionize the polygons to form one strip
		union_strip = unary_union(polys)
		#save it in a dictionary indexed by position coordinate
		if(direc == 'x'):
			self.x_strip_polys[pos] = union_strip
		elif(direc == 'y'):
			self.y_strip_polys[pos] = union_strip

			
	def build_tile(self):
		#loop over the x positions of y strips, 
		#and the y positions of x strips, building
		#strips at each point until reaching the desired N. 
		for n in range(self.N):
			pos = n*self.p + self.p/2.0
			self.build_strip(pos, 'x')
			self.build_strip(pos, 'y')
		
		#shift all of the polygons so that the tile outline
		#has 0,0 coordinate at the corner
		self.shift_all_polygons([self.tile_rim, self.tile_rim])


		#build the tile rim polygon
		coords = [np.array([0, 0]), \
					np.array([0, self.tile_length]), \
					np.array([self.tile_length, self.tile_length]), \
					np.array([self.tile_length, 0])]
		self.tile_outline = Polygon(coords)

	def shift_all_polygons(self, shift):
		self.polygons = [shapely.affinity.translate(_, shift[0], shift[1]) for _ in self.polygons] #list of all, non-unionized polygons
		self.pad_polys = [shapely.affinity.translate(_, shift[0], shift[1]) for _ in self.pad_polys] #List of all non-unionized pads only polygons
		new_x_polys = {}
		new_y_polys = {}
		for pos, p in self.x_strip_polys.items():
			p = shapely.affinity.translate(p, shift[0], shift[1])
			new_x_polys[pos] = p 
		for pos, p in self.y_strip_polys.items():
			p = shapely.affinity.translate(p, shift[0], shift[1])
			new_y_polys[pos] = p 
		
		self.x_strip_polys = new_x_polys
		self.y_strip_polys = new_y_polys

		if(self.tile_outline != None):
			self.tile_outline = shapely.affinity.translate(self.tile_outline, shift[0], shift[1])


	#centers all polygons so that the unit cell
	#is in the center
	def center_tile(self):
		N = self.N 
		p = self.p 
		#odd: at N/2*pitch
		if(N % 2 == 1):
			center = 0.5*N*p 
		#even: the crossing is slightly
		#off center relative to the edges. 
		#becomes 0.5*N*p - 0.5*p
		else:
			center = 0.5*N*p - 0.5*p


		new_x_polys = {}
		new_y_polys = {}
		for pos, p in self.x_strip_polys.items():
			p = shapely.affinity.translate(p, -1*center, -1*center)
			new_x_polys[pos - center] = p 
		for pos, p in self.y_strip_polys.items():
			p = shapely.affinity.translate(p, -1*center, -1*center)
			new_y_polys[pos - center] = p 
		
		self.x_strip_polys = new_x_polys
		self.y_strip_polys = new_y_polys
		
		

	def plot_polygons(self, polys=None, ax=None, show=True):
		if(ax is None):
			fig, ax = plt.subplots()

		if(polys is None):
			polys = self.polygons
			if(self.tile_outline != None):
				polys.append(self.tile_outline)

		for p in polys:
			ax.plot(*p.exterior.xy)


		if(show):
			plt.show()

		return ax


	#plots only the 0 centered unit cell version of strips
	def plot_unit_cell(self, ax = None, show = True):
		N = self.N 
		p = self.p 
		#odd: at N/2*pitch
		if(N % 2 == 1):
			center = 0.5*N*p 
		#even: the crossing is slightly
		#off center relative to the edges. 
		#becomes 0.5*N*p - 0.5*p
		else:
			center = 0.5*N*p - 0.5*p

		if(ax is None):
			fig, ax = plt.subplots(figsize=(8, 8))
			ax.set_xlabel("x (mm)")
			ax.set_ylabel("y (mm)")
			ax.set_aspect('equal')

		for pos, p in self.x_strip_polys.items():
			p = shapely.affinity.translate(p, -1*center, -1*center)
			ax.plot(*p.exterior.xy, 'k', linewidth=0.4)
		for pos, p in self.y_strip_polys.items():
			p = shapely.affinity.translate(p, -1*center, -1*center)
			ax.plot(*p.exterior.xy, 'k', linewidth=0.4)
		
		if(show):
			plt.show()

		return ax


	#plots x and y strip lists of unioned polygons
	def plot_strips(self, ax = None, show = True):
		if(ax is None):
			fig, ax = plt.subplots(figsize=(8, 8))
			ax.set_xlabel("x (mm)")
			ax.set_ylabel("y (mm)")
			ax.set_aspect('equal')

		for pos, p in self.x_strip_polys.items():
			ax.plot(*p.exterior.xy, color='#009E73', linewidth=0.8)
		for pos, p in self.y_strip_polys.items():
			ax.plot(*p.exterior.xy, color='#0072B2', linewidth=0.8)

		#plot tile outline
		if(self.tile_outline != None):
			ax.plot(*self.tile_outline.exterior.xy, 'k', linewidth=1)
		
		if(show):
			plt.show()

		return ax
		#plt.savefig("plots/tile-building/"+str(len(self.x_strip_polys)).zfill(2)+".png", bbox_inches='tight')

	"""
	def export_strips(self, filename):
		x_polygons = list(self.x_strip_polys.values())
		y_polygons = list(self.y_strip_polys.values())


		
		#create a schema for a mapping
		schema = {
			'geometry': 'LineString',
			'id': 'int',
			'FID': 'int'
		}

		#write a Fiona shapefile
		with fiona.open(filename+"_x.shp", 'w', 'ESRI Shapefile', schema) as c:
			for i, poly in enumerate(x_polygons):
				c.write({
					'geometry': mapping(poly.boundary),
					'id': i,
					'FID': i
					})
		#write a Fiona shapefile
		with fiona.open(filename+"_y.shp", 'w', 'ESRI Shapefile', schema) as c:
			for i, poly in enumerate(y_polygons):
				c.write({
					'geometry': mapping(poly.boundary),
					'id': i,
					'FID': i
					})

		#open the shapefiles and write to DXF
		shp = fiona.open(filename+"_x.shp")
		geojson_dict = {"type": "FeatureCollection", "features": []}
		for f in shp:
			geojson_dict["features"].append(f)
		with open(filename+"_x.geojson", "w") as gjs:
			json.dump(geojson_dict, gjs)

		#open the shapefiles and write to DXF
		shp = fiona.open(filename+"_y.shp")
		geojson_dict = {"type": "FeatureCollection", "features": []}
		for f in shp:
			geojson_dict["features"].append(f)
		with open(filename+"_y.geojson", "w") as gjs:
			json.dump(geojson_dict, gjs)

		#convert the geojson to dxf with an installed program
		#that is part of the GDAL set of programs. Please google
		#GDAL and install, setup paths to have access to the
		#binary "ogr2ogr"
		process = subprocess.run(['ogr2ogr', '-f', 'DXF', filename+"_x.dxf", filename+"_x.geojson"])
		process = subprocess.run(['ogr2ogr', '-f', 'DXF', filename+"_y.dxf", filename+"_y.geojson"])

	"""
	

	def print_area_information(self):
		a_c, a_d = self.get_total_areas()
		print("Total conductor area: {:.3f} mm^2, Total dielectric area: {:.3f} mm^2, Sum: {:.3f} mm^2".format(a_c, a_d, a_c+a_d))
		a_i, a_strip, a_all = self.get_crossing_areas()
		print("Single crossing area: {:.4f} mm^2, Single strip crossing area total: {:.4f} mm^2, All strips crossing area: {:.4f} mm^2".format(a_i, a_strip, a_all))

		print("Dielectric represents {:0.3f}% of the area".format(a_d*100/(a_c+a_d)))

	#return the area of conductor and dielectric
	def get_total_areas(self):
		#area of conductor, sum areas of strips
		total_conductor_area = 0
		for key, p in self.x_strip_polys.items():
			total_conductor_area += p.area
		for key, p in self.y_strip_polys.items():
			total_conductor_area += p.area

		#find the area of dielectric by finding total
		#area of square containing the assembly of strips,
		#and subtracting the total_conductor_area

		#length of this square
		l = self.N*self.p 
		total_dielectric_area = l**2 - total_conductor_area
		return total_conductor_area, total_dielectric_area

	#get info on areas of wire-crossings on strips. 
	def get_crossing_areas(self):
		#individual crossings
		indiv_cross_area = self.bridge_wid**2
		#crossings of one strip with all strips
		one_strip_crossing = 0
		#all strip crossings
		all_strip_crossing = 0
		for keyx, px in self.x_strip_polys.items():
			temp = 0
			for keyy, py in self.y_strip_polys.items():
				temp += py.intersection(px).area
			if(one_strip_crossing == 0):
				one_strip_crossing = temp 

			all_strip_crossing += temp

		return indiv_cross_area, one_strip_crossing, all_strip_crossing

	def get_strip_perimeters(self):
		xperims = []
		yperims = []
		for keyx, px in self.x_strip_polys.items():
			xperims.append(px.length)
		for keyy, py in self.y_strip_polys.items():
			yperims.append(py.length)
		return xperims, yperims


