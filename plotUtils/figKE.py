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

###########################################################

def splitVel(u,v):

    tmpa, tmpb = x.getvrtdivspec(u,v)
    phi = x.invlap*tmpb
    udiv,vdiv = x.getgrad(phi)
    urot = u - udiv
    vrot = v - vdiv
    return[udiv,vdiv,urot,vrot]

###########################################################


##########################################################

def calcFlux(name,tm):

    data = xray.open_dataset(name)
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
    data.close()

    us = usel[0,:,:].values
    vs = vsel[0,:,:].values
    udiv,vdiv,urot,vrot = splitVel(us,vs)


#    usmean = np.mean(us,axis=1)*factor
#    urotmean = np.mean(urot,axis=1)*factor
#    udivmean = np.mean(udiv,axis=1)*factor

    return [us,udiv,urot,vs,vdiv,vrot]


########################################################

path = '/home/suhas/sphere_data/turbulence/vorticity_blob/'

filename = 'hbar=300,momentum=10,radiative=10,fr=1e-10_dlat=10_exp_dlon=30_exp_pair.nc'
flux = calcFlux(path+filename,100)

#path = '/home/suhas/sphere_data/gill/time_derivative/'
#filename = 'hbar=300,momentum=10,radiative=10,fr=1e-4_cos_exp.nc'
#flux = calcFlux(path+filename,0)






###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myCmap import *
from plotUtils import *


plt.ion()
fig,ax = plt.subplots(1,3, sharey='all',figsize=(10,5))
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})

longi = np.rad2deg(longi)
lati = np.rad2deg(lati)

[u,udiv,urot,v,vdiv,vrot] = flux

KE = np.mean(0.5*(u**2 + v**2),axis=1)*(factor**2)
KE_div = np.mean(0.5*(udiv**2 + vdiv**2),axis=1)*(factor**2)
KE_rot = np.mean(0.5*(urot**2 + vrot**2),axis=1)*(factor**2)

lati = lati[:,0]

ax[0].plot(KE,lati)
ax[1].plot(KE_div,lati)
ax[2].plot(KE_rot,lati)



for i in range(3):
    ax[i].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
    ax[i].ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
    ax[i].set_xlabel(u'm$\mathregular{^{2}}$s$\mathregular{^{\u22122}}$',size='12', fontname = 'Dejavu Sans')

ax[0].set_title('KE',size='12', fontname = 'Dejavu Sans')
ax[1].set_title('KE_div',size='12', fontname = 'Dejavu Sans')
ax[2].set_title('KE_rot',size='12', fontname = 'Dejavu Sans')

ax[0].set_ylim(-60,60)

ax[0].text(0.03, 0.97, '(a)', fontname = 'Dejavu Sans', transform=ax[0].transAxes, fontsize=12, va='top')
ax[1].text(0.03, 0.97, '(b)', fontname = 'Dejavu Sans', transform=ax[1].transAxes, fontsize=12, va='top')
ax[2].text(0.03, 0.97, '(c)', fontname = 'Dejavu Sans', transform=ax[2].transAxes, fontsize=12, va='top')

ax[0].set_ylabel(u'Latitude (\u00B0)',size='12', fontname = 'Dejavu Sans')

#plt.savefig('/home/suhas/Dropbox/SWE_Rev/KE_split.pdf', bbox_inches='tight',dpi=600)
plt.savefig('/home/suhas/Dropbox/SWE_final/Figure4.pdf', bbox_inches='tight',dpi=600)
##############################################


