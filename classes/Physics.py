import numpy as np
from scipy.interpolate import interp1d

#transverse diffusion as function of field.
def calc_trans_diffusion(field):
	#E = [20, 38, 75, 187, 376.0, 615] #V/cm
	#coef = [26.5, 42, 48.5, 54.0, 53, 60] #cm^2/sec
	#coef = [_*1e-4 for _ in coef] #mm^2/us
	#f = interpolate.interp1d(E, coef)
	#from exo-200 data
	coef = 55 #cm^2/s
	coef = coef*1e-4 #mm^2/us
	return coef

#longitudinal diffusion as function of field.
def calc_long_diffusion(field):
	E = [60, 70, 80, 90, 100, 200, 300, 400, 500, 750, 1000] #V/cm
	coef = [69.6, 64.1, 61.7, 54, 46.7, 31.1, 26.7, 24.3, 24.1, 22.9, 21.4] #cm^2/sec
	coef = [_*1e-4 for _ in coef] #mm^2/us
	f = interp1d(E, coef)
	return f(field)

def calc_drift_velocity(field):
	#put in the actual table of velocities later
	return 1.7 #mm/us

def sample_initial_charge_radius():
	cdf_file = '../data/diameter_vs_cdf.p' #generated by Evan using nexo-offline 0vbb's
	s = pickle.load(open(cdf_file, 'rb'))[0] #an interp1d object with CDF value as input and diameter as output

	#monte-carlo on the CDF with one sample
	c = np.random.uniform(0, 1, 1)[0] #one value
	diam = s(c)
	return diam/2



