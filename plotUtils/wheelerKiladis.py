#==================================================================
### Borrowed from ncarg/nclscripts/csm/diagnostics_cam.ncl

# Note_1: The full logitudinal domain is used.
#         This means that every planetary
#         wavenumber will be represented.
# Note_2: Tapering in time is done to make the variable periodic.
#
# The calculations are also only made for the latitudes
# between '-latBound' and 'latBound'.
#####################################################################

import numpy as np
import xarray
from numpy.fft import fft,fft2,fftshift, fftfreq

def wk(var,latmin,latmax,tmin,tmax,dt,dx=1.):
    sel = {}
    sel['latitude']  = slice(latmax,latmin)  # symmetric about equator ?
    sel['time'] = slice(tmin,tmax)      # For some time period specified
    data = var.loc[sel]
#    data = data - data.mean(dim=('time'))
    data_rev = (data.sel(latitude=slice(None, None, -1))).values # Flip along lat axis 
    data_sym  = (data + data_rev) /2.
    data_asym = (data - data_rev) /2.

    power  = calcPower(data)
    pwsym  = calcPower(data_sym)
    pwasym = calcPower(data_asym)


    ### Define time,lat and lon from data
    time = data.time
    lon = data.longitude
    ntim = len(time)
    nlon = len(lon)
    dx = dx/nlon
    kxx = fftfreq(nlon,dx)
    ktt = fftfreq(ntim,dt)
    [kx,kt] = np.meshgrid(kxx,ktt)  
    kx = fftshift(kx)
    kt = fftshift(kt)
    power,pwsym,pwasym = apply_filter(power,pwsym,pwasym,kx,kt) 
    return [power,pwsym,pwasym,kx,kt]

def calcPower(data):
    p  = fft2(data,axes=(0,2))
    power = (p*p.conj()).real
    power = np.mean(power,axis=1)
    #    power = power/np.amax(abs(power))
    power = fftshift(power)
    return np.flip(power,axis=0)

def filter121(data):
    weights = np.array((0.25,0.5,0.25))
    mavg = np.convolve(data,weights,'same')
    return mavg   
   
#######################################################################################
def apply_filter(dtot,dsym,dasym,kx,kt):
    kx = kx[0,:]
    kt = kt[:,0]
    ### Filtering the sym and asym parts
    ### Only freq filtering
    for i in range(5):
        dsym[:,(abs(kx)<=27.)]  = np.apply_along_axis(filter121,0,dsym[:,(abs(kx)<=27.)])
        dasym[:,(abs(kx)<=27.)] = np.apply_along_axis(filter121,0,dasym[:,(abs(kx)<=27.)])

    ############ Background spectra ##########
    #### Smoothen it a lot !!!!!!!
    for j in range(5):
        idx = (abs(kt)<= 0.1)
        dtot[idx,:]  = np.apply_along_axis(filter121,1,dtot[idx,:])
    for j in range(10):
        idx = np.logical_and(abs(kt)>0.1,abs(kt)<=0.2)
        dtot[idx,:]  = np.apply_along_axis(filter121,1,dtot[idx,:])
    for j in range(20):
        idx = np.logical_and(abs(kt)>0.2,abs(kt)<=0.3)
        dtot[idx,:]  = np.apply_along_axis(filter121,1,dtot[idx,:])
    for j in range(40):
        idx = (abs(kt) > 0.3)
        dtot[idx,:]  = np.apply_along_axis(filter121,1,dtot[idx,:])
    for j in range(10):
#        idx = (abs(kt)<=0.8)
        jdx = (abs(kx)<=27.)
        dtot[:,jdx]  = np.apply_along_axis(filter121,0,dtot[:,jdx])
    return dtot,dsym,dasym

    


