#Code to solve moist shallow water equations on a sphere
#SWE are of the vorticity-divergence form.
import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import scipy as sc
import xarray
import netCDF4
import seaborn

seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
#plt.ion()
###########################################

hres_scaling=4

nlons  = 512  # number of longitudes
nlons  = 2*512  # number of longitudes
nlons  = hres_scaling*512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)   # number of lats for gaussian grid.

# parameters for test
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity
Hbar    = 300.      # mean height (some typical range)

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lats) # coriolis

# Relaxation time scales
tau_vrt = 10*24*3600.
tau_div = 10*24*3600.
tau_ht = 10*24*3600.

tau_q   = 10*24*3600.   # evaporation time scale
tau_c   = 0.1*24*3600.  # condensation time scale

chi = 5500.  #### coversion factor similar to latent heat
# make sure that (h - chi*Qsat) is always positive

latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)

##################################################################
# setting up the integration
tmax = 10.01*24*3600  #(time to integrate, here 10 days)
tmax = 15.01*24*3600  #(time to integrate, here 15 days)
tmax = 30.01*24*3600  #(time to integrate, here 30 days)
################################################################
latdim = nlats
londim = nlons
dims = [latdim,londim]
dim_names = ['latitude','longitude']

path     = './'  # path to save the file
filename = 'testMoist_smallOutputTimestep' # filename
filename = 'testMoist_chi1500'
filename = 'testMoist_chi2500'
filename = 'testMoist_chi3500'
filename = 'testMoist_chi4500'
filename = 'testMoist' # filename


filename = 'testMoist_smallOutputTimestep_dt100s'

fields   = ['uwind','vwind','vorticity','divergence','height','qprime']
 # variables to store (this only saves the data for a field which is a fn of lat, lon and time).
cords    = [x.lats,x.lons]

const_name = ['radius','rotation','gravity', 'chi'] #for constants
const_data = [rsphere,omega,grav, chi]



dataset = netCDF4.Dataset(filename+'.nc')

var='vorticity'
var='divergence'
var='height'
var='qprime'
#div=dataset.variables['divergence'][it,:,:]
#vort=dataset.variables['vorticity'][it,:,:]
#u=dataset.variables['uwind'][it,:,:]
#v=dataset.variables['vwind'][it,:,:]
#q=dataset.variables['qprime'][it,:,:]
#ht=dataset.variables['height'][it,:,:]


field3d=dataset.variables[var]
timeseries=dataset.variables[var][:,0,0]*0.
nt=timeseries.shape[0]
for it in range(0,nt):
    timeseries[it]=np.mean(field3d[it,:,:]); metric='mean'
    #timeseries[it]=np.mean(np.abs(field3d[it,:,:])); metric='meanabs'
time=np.arange(0,tmax,tmax/nt)/3600./24.
print('time ', time)

plt.figure(1,figsize=(8,4))
plt.clf()
plt.plot(time,timeseries)
plt.title(var)
plt.xlabel('time / days')
plt.grid()
plt.savefig(var+'_'+metric+'_'+filename+'_timeseries')

np.save(var+'_'+metric+'_'+filename+'_timeseries', timeseries)
np.save(var+'_'+metric+'_'+filename+'_timeseries_time', time)
