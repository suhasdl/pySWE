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
from numpy.fft import fft, ifft, fft2,ifft2,fftshift,fftfreq
import shtns
import sphTrans as sph
import matplotlib.pyplot as plt
import spectra
import seaborn
import dispersionCurves as dc


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
Hbar = 300.

def calcFlux(name,tmin,tmax,dt,dx=1.):

    data = xarray.open_dataset(name)
    u    = data['uwind']
    vort = data['vorticity']
    ht   = data['height']
#    qp   = data['qprime']
    vrt  = data['vorticity']

    sel={}
    sel['time'] = slice(tmin,tmax)
    sel['latitude'] = slice(np.deg2rad(15),np.deg2rad(-15))
    vrtsel = vrt.loc[sel]
    htsel = ht.loc[sel]
    out   = htsel 
#    out   = vrtsel
    out   = out - out.mean(dim=('time'))
    power   = calcPower(out)

    ##########################################
    lat  = out.latitude
    lon  = out.longitude
    time = out.time
    ntim = len(time)
    nlon = len(lon)
    dx   = dx/nlon
    kxx  = fftfreq(nlon,dx)
    ktt  = fftfreq(ntim,dt)
    [kx,kt] = np.meshgrid(kxx,ktt)
    kx   = fftshift(kx)
    kt   = fftshift(kt)
    # power[abs(kx)<1] = 0.
    power = power/np.amax(abs(power))
    power = np.log10(power)
    return [power,kx,kt]

def calcPower(data):
    p     = fft2(data,axes=(0,2))
    power = (p*p.conj()).real
    power = np.mean(power,axis=1)
#    power = power/np.amax(abs(power))
    power = fftshift(power)
    return np.flip(power,axis=0)
########################################################

tmin = 500.
tmax = 1000.
dt   = 1.

path     = '/media/suhas/Data/moist/SR/longRun/'

filename = 'IVP,Hbar=300,vrtamp=1e-6_matsuno,chi=5500,Qbar=50e-3_lat,q=qsat_5deg,tau=Inf,tauq=Inf,tauc=1e-1.nc'
spby10   = calcFlux(path+filename,tmin,tmax,dt)

filename = 'IVP,Hbar=300,vrtamp=1e-6_matsuno,chi=5500,Qbar=50e-3_lat,q=qsat_5deg,tau=Inf,tauq=Inf,tauc=1.nc'
sp1     = calcFlux(path+filename,tmin,tmax,dt)

filename = 'IVP,Hbar=300,vrtamp=1e-6_matsuno,chi=5500,Qbar=50e-3_lat,q=qsat_5deg,tau=Inf,tauq=Inf,tauc=10.nc'
sp10 = calcFlux(path+filename,tmin,tmax,dt)

filename = 'IVP,Dry,Hbar=300,vrtamp=1e-6_matsuno,tau=Inf.nc'
spdry = calcFlux(path+filename,tmin,tmax,dt)



###########################################################################
from mpl_toolkits.axes_grid1 import make_axes_locatable
from myCmap import *
from plotUtils import *
plt.ion()
#seaborn.set_context("talk")
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})


fig,ax = plt.subplots(2,2, figsize=(15, 12))

disp = dc.curves(lat=np.deg2rad(7.5), grav=grav,rsphere=rsphere,omega=omega,mean=0.)
##  KW ###

def plotDispCurves(Hbar,i,j,color):
    freqKW,kKW   = disp.KW(Hbar)
    freqRW1,kRW1 = disp.ERW(Hbar,1.)
    freqRW2,kRW2 = disp.ERW(Hbar,2.)
    freqRW3,kRW3 = disp.ERW(Hbar,3.)
    freqMRG,kMRG = disp.MRG(Hbar)

    freqIGW1,kIGW1 = disp.IGW(Hbar,1.)
    freqIGW2,kIGW2 = disp.IGW(Hbar,2.)
    freqIGW3,kIGW3 = disp.IGW(Hbar,3.)
    freqIGW4,kIGW4 = disp.IGW(Hbar,4.)
    freqIGW5,kIGW5 = disp.IGW(Hbar,5.)

    ax[i,j].plot(kKW ,freqKW,color=color)
    ax[i,j].plot(kRW1,freqRW1,color=color)
    ax[i,j].plot(kRW2,freqRW2,color=color)
    ax[i,j].plot(kRW3,freqRW3,color=color)
    ax[i,j].plot(kMRG,freqMRG,color=color)
    ax[i,j].plot(kIGW1,freqIGW1,color=color)
    ax[i,j].plot(kIGW2,freqIGW2,color=color)
    ax[i,j].plot(kIGW3,freqIGW3,color=color)
    ax[i,j].plot(kIGW4,freqIGW4,color=color)
    ax[i,j].plot(kIGW5,freqIGW5,color=color)



def plotFig(figName,i,j,Hbar,color='k'):
    [power,kx,kt] = figName
    levels = np.linspace(-3,0,15.)
    contours = ax[i,j].contourf(kx,kt,power,levels=levels,cmap='Spectral_r',extend='both',zorder=0)
    cb = plt.colorbar(contours,ax=ax[i,j], orientation='horizontal',shrink=0.95,aspect=30,pad=0.14)
    plt.ticklabel_format(style='sci',scilimits=(-2,2),axis='x')
    plt.ticklabel_format(style='sci',scilimits=(-2,2),axis='y')
    ax[i,j].set_ylim(0,0.5)
    ax[i,j].set_xlim(-10,10)
    ax[i,j].axvline(x=0,linestyle='--',color='k')
    ax[i,j].set_xlabel('kx ',size='12', fontname = 'Dejavu Sans')
    ax[i,j].set_ylabel('freq (day^-1)',size='12', fontname = 'Dejavu Sans')
    plotDispCurves(Hbar,i,j,color)
    ax[i,j].grid(alpha=0.7,color='k',linestyle='dotted',dashes=[1,5 ],linewidth=1,zorder=100)
##################################################
plotFig(spby10,0,0,300.)
plotFig(sp1,0,1,300.)
plotFig(sp10,1,0,300.)
plotFig(spdry,1,1,300.)

ax[0,0].set_title('tauc = 0.1',size='12', fontname = 'Dejavu Sans')
ax[0,1].set_title('tauc = 1'  ,size='12', fontname = 'Dejavu Sans')
ax[1,0].set_title('tauc = 10',size='12', fontname = 'Dejavu Sans')
ax[1,1].set_title('dry'  ,size='12', fontname = 'Dejavu Sans')


plt.suptitle("Wheeler-Kiladis (Day 500-1000) \n IVP vorticity forcing (Matsuno like)",size='14', fontname = 'Dejavu Sans')
plt.savefig('/home/suhas/Desktop/IVPwk.pdf', bbox_inches='tight',dpi=150)
