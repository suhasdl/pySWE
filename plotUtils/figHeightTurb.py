import numpy as np
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import time
import AdamsBashforth
import sphericalForcing as spf
import scipy as sc
import xarray
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


################################################

def calcFlux(name,tm):

    data = xarray.open_dataset(name)
    lats = data.latitude.copy()
    lons = data.longitude.copy()
    u = data.uwind.copy()
    v = data.vwind.copy()
    ht = data.height.copy()

    sel={}
    sel['time'] = slice(tm,tm)

    usel = u.loc[sel]
    vsel = v.loc[sel]
    htsel = ht.loc[sel]

    uprime = usel - usel.mean(dim='longitude')
    vprime = vsel - vsel.mean(dim='longitude')
    htprime = htsel - htsel.mean(dim='longitude')

    up = uprime.mean(dim='time')
    vp = vprime.mean(dim='time')
    htp = htprime.mean(dim='time')

    return [htp.values,up.values,vp.values]


########################################################
path = '/home/suhas/sphere_data/turbulence/forcing_timederiv/'

filename = 'hbar=300,momentum=100,radiative=100,kf=100,fr=5_20N-20S,corr=0,ZARemoved,longrun.nc'

flux = calcFlux(path+filename,2000)








###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myCmap import *
from plotUtils import *
plt.ion()
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})

figName = flux
fig = plt.figure()
ax = fig.gca()

longi = np.rad2deg(longi)
lati = np.rad2deg(lati)

plotLevels, plotTicks = getLevelsAndTicks(np.amax(abs(figName[0])),levels=16)
contours=plt.contourf(longi,lati,figName[0],levels=plotLevels, cmap=joyDivCmapRdBl,extend='both',zorder=0)
cb = plt.colorbar(contours, orientation='horizontal',shrink=0.6,aspect=40,pad=0.12)
cb.set_ticks(plotTicks)
cb.ax.set_xlabel('m',size='12', fontname = 'Dejavu Sans')
plt.grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
k = 8
l = 20
Qp = plt.quiver(longi[::k,::l],lati[::k,::l],figName[1][::k,::l],figName[2][::k,::l],scale=400,width=0.0012,zorder=20)
plt.quiverkey(Qp, 0.9, -0.175, 15, u'15 ms$\mathregular{^{\u22121}}$',fontproperties={'weight':'roman','size':'12'})
ax.set_xticks(np.linspace(0,360,9))
ax.set_yticks(np.linspace(-90,90,7))

plt.ylabel(u'Latitude (\u00B0)',size='12', fontname = 'Dejavu Sans')
plt.xlabel(u'Longitude (\u00B0)',size='12', fontname = 'Dejavu Sans')
   


#plt.savefig('/home/suhas/Dropbox/SWE_Rev/height_turb.pdf', bbox_inches='tight',dpi=600)
plt.savefig('/home/suhas/Dropbox/SWE_final/Figure10.pdf', bbox_inches='tight',dpi=600)
