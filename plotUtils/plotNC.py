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
nlons  = 512  # number of longitudes
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
filename = 'testMoist' # filename
#filename = 'testDryGill' # filename

fields   = ['uwind','vwind','vorticity','divergence','height','qprime']
 # variables to store (this only saves the data for a field which is a fn of lat, lon and time).
cords    = [x.lats,x.lons]

const_name = ['radius','rotation','gravity', 'chi'] #for constants
const_data = [rsphere,omega,grav, chi]



dataset = netCDF4.Dataset(filename+'.nc')
it=28
it=22
it=14
#it=13
#it=12
#it=11
#it=10
#it= 9
#it= 8
#it= 7
#it= 6
#it= 5
#it= 4
#it= 3
#it= 2
#it= 1
div=dataset.variables['divergence'][it,:,:]
vort=dataset.variables['vorticity'][it,:,:]
u=dataset.variables['uwind'][it,:,:]
v=dataset.variables['vwind'][it,:,:]
q=dataset.variables['qprime'][it,:,:]
ht=dataset.variables['height'][it,:,:]

print("ht.shape", ht.shape)
print("lonDeg.shape", lonDeg.shape)
print("latDeg.shape", latDeg.shape)

speed=np.sqrt(u*u + v*v)

lw = 5*speed / speed.max()
lw = 2*speed / speed.max()
lw = np.sqrt(speed / speed.max())


xedge=70
yedge=60
u=u[yedge:nlats-yedge,xedge:nlons]
v=v[yedge:nlats-yedge,xedge:nlons]
ht=ht[yedge:nlats-yedge,xedge:nlons]

div=div[yedge:nlats-yedge,xedge:nlons]
vort=vort[yedge:nlats-yedge,xedge:nlons]
q=q[yedge:nlats-yedge,xedge:nlons]

lonDeg=lonDeg[yedge:nlats-yedge,xedge:nlons]
latDeg=latDeg[yedge:nlats-yedge,xedge:nlons]
lw=lw[yedge:nlats-yedge,xedge:nlons]

plt.figure(1,figsize=(8,4))
plt.clf()
maxAnomaly = np.amax(abs(q))
if maxAnomaly == 0 : maxAnomaly = 0.0001
maxAnomaly = 8.
dtick = 1.
qticks=np.arange(-maxAnomaly, maxAnomaly,dtick)
qlevels = np.linspace(-maxAnomaly, maxAnomaly, 100)
plt.contourf(lonDeg,latDeg,q*1.e3,qlevels,extend='both',cmap='coolwarm')
plt.colorbar(orientation='horizontal',extend="both",ticks=qticks)
maxAnomaly = np.amax(abs(div))
print("maxDIV: ",maxAnomaly)
maxAnomaly = 3 
levels = np.linspace(-maxAnomaly, maxAnomaly, 7)
levels = [-4, -3, -2, -1, -.5, .5, 1, 2, 3, 4]
levels = [-4, -3, -2, -1, 1, 2, 3, 4]
cs=plt.contour(lonDeg,latDeg,div*1.e7,levels,colors='k',linewidths=.5)
plt.clabel(cs, fontsize=9, inline=1)
plt.title('q')
# plt.contour(lonDeg, latDeg, Qbar,6,colors='r')
plt.grid()
#plt.pause(1e-3)
plt.savefig('q_'+filename+'_'+str(it).zfill(2))


plt.figure(2,figsize=(8,4))
plt.clf()
maxAnomaly = np.amax(abs(ht-300))
maxAnomaly = 2.8
dtick=.4
levels = np.linspace(-maxAnomaly, maxAnomaly, 100)
#plt.contourf(lonDeg,latDeg,ht-300.,levels,extend='both',cmap='coolwarm')
#plt.contourf(lonDeg,latDeg,ht-300.,levels,extend='both',cmap='Greys')
#plt.contourf(lonDeg[yedge:nlats-yedge,xedge:nlons],latDeg[yedge:nlats-yedge,xedge:nlons],ht[yedge:nlats-yedge,xedge:nlons]-300.,levels,extend='both',cmap='RdBu_r')
plt.contourf(lonDeg,latDeg,ht-300.,levels,extend='both',cmap='RdBu_r')
plt.colorbar(orientation='horizontal',extend="both",ticks=np.arange(-maxAnomaly, maxAnomaly,dtick))
#plt.streamplot(lonDeg[yedge:nlats-yedge,xedge:nlons], latDeg[yedge:nlats-yedge,xedge:nlons], u[yedge:nlats-yedge,xedge:nlons], v[yedge:nlats-yedge,xedge:nlons], density=0.8, color='k', linewidth=lw[yedge:nlats-yedge,xedge:nlons])
plt.streamplot(lonDeg, latDeg, u, v, density=0.8, color='k', linewidth=lw)
plt.title('h')
# plt.contour(lonDeg, latDeg, Qbar,6,colors='r')
plt.grid()
plt.savefig('ht_'+filename+'_'+str(it).zfill(2))
