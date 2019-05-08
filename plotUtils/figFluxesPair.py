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
factor = np.cos(x.lats)
Hbar = 300.
ndiss = 8.
diff_fact = (1e-5*((x.lap/x.lap[-1])**(ndiss/2)))
l = x._shtns.l
###########################################################
f = 2.*omega*np.sin(lati)




def yDeriv(data):

    a1 = np.tile(data,(512,1))
    a2 = a1.transpose()  
    qs = x.grdtospec(a2)
    a,out = x.getgrad(qs)
    return out[:,0]

def yDeriv2D(data):
    a = x.grdtospec(data)
    b,out = x.getgrad(a)
    return out

def xDeriv2D(data):
    a = x.grdtospec(data)
    out,b = x.getgrad(a)
    return out


################################################
def mean_product(a,b):
    a1 = a - np.mean(a,axis=1)[:,np.newaxis]
    b1 = b - np.mean(b,axis=1)[:,np.newaxis]
    prod = a1*b1
    flux = np.mean(prod,axis=1)
    return flux
#################################################

def moving_avg(data,window):
    weights = np.repeat(1.0,window)/window
    mavg = np.convolve(data,weights,'valid')
    return mavg

#################################################

def calcFlux(name,tm,frot=f,tau_m=10*24*3600.,tau_r=10*24*3600.):

    print name
    
    data = xray.open_dataset(name)
    lats = data.latitude.copy()
    lons = data.longitude.copy()
    time = data.time.copy()
    u = data.uwind.copy()
    v = data.vwind.copy()
    ht = data.height.copy()
    vrt = data.vorticity.copy()
    uhp_dt = data.uhpdt.copy()
    Fr = data.forcing.copy()

    tsel={}
    tsel['time'] = slice(tm,tm)

    usel = u.loc[tsel]
    vsel = v.loc[tsel]
    htsel= ht.loc[tsel]
    vrtsel = vrt.loc[tsel]
    uhp_dtsel = uhp_dt.loc[tsel]

    data.close()

    for i in np.arange(0,usel.shape[0]):
        print i
        us = usel[i,:,:].values
        vs = vsel[i,:,:].values
        hts = htsel[i,:,:].values
        vrts = vrtsel[i,:,:].values
        uhp_dts = uhp_dtsel[i,:,:].values
        Frs = Fr[i,:,:].values
        Frsf = x.spectogrd(x.invlap*x.grdtospec(Frs))
        S = -yDeriv2D(Frsf)


        hbar = np.mean(hts,axis=1)
        vbar = np.mean(vs,axis=1)
        ubar = np.mean(us,axis=1)
        vhbar = np.mean(hts*vs,axis=1)
        vrtabar = np.mean(vrts+frot, axis=1)
        uhbar = np.mean(us*hts,axis=1)
        Sbar = np.mean(S,axis=1)


 
        uvp = mean_product(us,vs)
        uhp = mean_product(us,hts)
        usp = mean_product(us,S)
        vhp = mean_product(vs,hts)
        hsp = mean_product(hts,S)



        eddy_flux = -yDeriv((hbar*uvp + vbar*uhp))/hbar


        radiative_flux =  -uhp/(hbar*tau_r)
        source_flux = Sbar + hsp/hbar
        drag1_flux = -uhp/(tau_m*hbar)
        drag2_flux = -ubar/(tau_m)


        vorticity_flux = vhbar*vrtabar/hbar
        time_flux = -uhp_dts[:,1]/hbar






        total_flux = eddy_flux + radiative_flux + source_flux + vorticity_flux + drag1_flux + drag2_flux + time_flux
    
    u_zon  = ubar*factor


    return [eddy_flux , source_flux , radiative_flux ,drag1_flux , drag2_flux ,vorticity_flux , time_flux ,total_flux , u_zon ]




########################################################
lats = x.lats

path = '/home/suhas/sphere_data/turbulence/vorticity_blob/'


filename = 'hbar=300,momentum=10,radiative=10,fr=1e-10_dlat=10_exp_dlon=30_exp_pair.nc'

flux = calcFlux(path+filename,100)


flux_ini = calcFlux(path+filename,2)

###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable



seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
plt.ion()
fig,ax = plt.subplots(1,5,sharey='all',figsize=(15,5))
#fig.tight_layout(w_pad=1, h_pad=3)
#fig.tight_layout(pad=0.5, w_pad=0.5, h_pad=0.5)
#fig.subplots_adjust(hspace=0.25)
lats = np.rad2deg(lats)
def plotFig(name,style,clr='#4c72b0'):
    ax[0].plot(name[0],lats,linewidth=2,linestyle=style,color=clr)
    ax[1].plot(name[1],lats,linewidth=2,linestyle=style,color=clr)
    ax[2].plot(name[3]+name[4],lats,linewidth=2,linestyle=style,color=clr)
    ax[3].plot(name[5],lats,linewidth=2,linestyle=style,color=clr)
    ax[4].plot(name[7],lats,linewidth=2,linestyle=style,color=clr)
        

        

plotFig(flux_ini,'solid')
plotFig(flux,'dashed')

from matplotlib.ticker import FuncFormatter

for i in range(5):
        ax[i].ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
        ax[i].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=10)
        xmin,xmax = ax[i].get_xlim()
        xxx = np.linspace(xmin,xmax,5)
        ax[i].set_xticks(xxx)
        ax[i].axvline(x=0,color='k',linewidth=1.1,zorder=25)
        ax[i].set_yticks(np.linspace(-60,60,5))
        ax[i].set_ylim([-60,60])
        ax[i].set_xlabel(u'ms$\mathregular{^{\u22122}}$',size='12', fontname = 'Dejavu Sans')


ax[0].set_title('Eddy', size='12', fontname = 'Dejavu Sans')
ax[1].set_title('Momentum source',size='12', fontname = 'Dejavu Sans')
ax[2].set_title('Momentum drag',size='12', fontname = 'Dejavu Sans')
ax[3].set_title('Vorticity',size='12', fontname = 'Dejavu Sans')
ax[4].set_title('Total flux',size='12', fontname = 'Dejavu Sans')

ax[0].set_ylabel(u'Latitude (\u00B0)',size='12', fontname = 'Dejavu Sans')

ax[0].text(0.03, 0.97, '(a)', fontname = 'Dejavu Sans', transform=ax[0].transAxes, fontsize=12, va='top')
ax[1].text(0.03, 0.97, '(b)', fontname = 'Dejavu Sans', transform=ax[1].transAxes, fontsize=12, va='top')
ax[2].text(0.03, 0.97, '(c)', fontname = 'Dejavu Sans', transform=ax[2].transAxes, fontsize=12, va='top')
ax[3].text(0.032, 0.97, '(d)', fontname = 'Dejavu Sans', transform=ax[3].transAxes, fontsize=12, va='top',backgroundcolor='w',zorder=50)
ax[4].text(0.03, 0.97, '(e)', fontname = 'Dejavu Sans', transform=ax[4].transAxes, fontsize=12, va='top')


#plt.savefig('/home/suhas/Dropbox/SWE_Rev/flux_pair.pdf', bbox_inches='tight',dpi=600)
plt.savefig('/home/suhas/Dropbox/SWE_final/Figure2.pdf', bbox_inches='tight',dpi=600)

