import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import AdamsBashforth
import sphericalForcing as spf
import scipy as sc
import xray
import logData
import netCDF4
import matplotlib.ticker as mticker
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
from numpy.fft import fft, ifft, fft2,ifft2
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import spectra
import seaborn

nlons = 512  # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats = int(nlons/2)   # for gaussian grid.
 
# parameters for test
rsphere = 6.37122e6 # earth radius
omega = 7.292e-5 # rotation rate
grav = 9.80616 # gravity

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
longi,lati = np.meshgrid(x.lons, x.lats)
f = 2.*omega*np.sin(lati)
factor = np.cos(x.lats)

l = x._shtns.l
##################################################################
def energy(u,v):

    uk = x.grdtospec(u)
    vk = x.grdtospec(v)
    Esp = 0.5*(uk*uk.conj()+vk*vk.conj())

    Ek = np.zeros(np.amax(l)+1)
    k = np.arange(np.amax(l)+1)
    for i in range(0,np.amax(l)):
        Ek[i] = np.sum(Esp[np.logical_and(l>=i-0.5 , l<i+0.5)])

    return Ek,k
####################################################################

def calcFlux(name,tmin,tmax):

    data = xray.open_dataset(name)
    lats = data.latitude.copy()
    lons = data.longitude.copy()
    u = data.uwind.copy()
    v = data.vwind.copy()
    ht = data.height.copy()

    sel={}
    sel['time'] = slice(tmin,tmax)

    usel = u.loc[sel]
    vsel = v.loc[sel]

    data.close()

    for i in np.arange(0,usel.shape[0]):
        print i
        us = usel[i,:,:].values
        vs = vsel[i,:,:].values
        Ek,k = energy(us,vs)

        if i==0:
            Ek_stack = Ek
        else:
            Ek_stack = np.vstack((Ek_stack,Ek))
            
    Ek_mean = np.mean(Ek_stack,axis=0)

    return [Ek_mean,k]


########################################################
path = '/home/suhas/sphere_data/turbulence/forcing_timederiv/'

filename = 'hbar=300,momentum=100,radiative=100,kf=100,fr=5_20N-20S,corr=0,ZARemoved,longrun.nc'

flux = calcFlux(path+filename,500,2000)





def logline(x1,y1,x2,m):
    y2 = 10**(m*np.log10(x2/x1)+np.log10(y1))
    plt.plot([x1,x2],[y1,y2],color='k')



###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myCmap import *
from plotUtils import *
plt.ion()
#fig,ax = plt.subplots(2,3,sharex='all', sharey='all',figsize=(15, 10))

seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})

plt.figure(figsize=(6,5))
plt.loglog(flux[1],flux[0],linewidth=2)
logline(17.,1.3,80.,-5/3.)
plt.xlabel('k',size='12', fontname = 'Dejavu Sans')
plt.ylabel('E(k)',size='12', fontname = 'Dejavu Sans')
plt.grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
plt.xlim(0,160)
#plt.savefig('/home/suhas/Dropbox/SWE_Rev/spectra.pdf', bbox_inches='tight',dpi=600)
plt.savefig('/home/suhas/Desktop/spectra.pdf', bbox_inches='tight',dpi=600)

