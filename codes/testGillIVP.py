"""
Code to solve shallow water equations on a sphere
SWE are of the vorticity-divergence form
Similar to the Gill problem but the system is only forced at t = 0 and then allowed to evolve freely.
I have also switched off the damping.
"""
from __future__ import print_function
import numpy as np
import sphTrans as sph
import matplotlib.pyplot as plt
import AdamsBashforth
import logData
import seaborn
seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
plt.ion()
###########################################

nlons  = 512           # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)  # for gaussian grid

# parameters for the simulations
rsphere = 6.37122e6 # earth radius
omega   = 7.292e-5  # rotation rate
grav    = 9.80616   # gravity

# setup up spherical harmonic instance, set lats/lons of grid
x = sph.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats) # meshgrid
f = 2.*omega*np.sin(lats) # coriolis
latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)

# Relaxation time scales
tau_vrt = np.Inf # no damping
tau_div = np.Inf
tau_ht  = np.Inf

########### Mass forcing #########################################
# a single gaussian blob for gill forcing
# change it to any other config if so desired
lats_ref   = np.radians(0.)   # lat centre
lons_ref   = np.radians(180.) # lon centre
delta_lats = np.radians(10.)  # lat extent
delta_lons = np.radians(30.)  # lon extent
hamp = 10. # Initial amplitude
S = hamp * np.exp(-((lons-lons_ref)/(delta_lons))**2) * np.exp(-((lats-lats_ref)/(delta_lats))**2)
Hbar = 300. # mean height

# Initialise other fields
zero_array = np.zeros_like(S)
htsp = x.grdtospec(S + Hbar)
zero_array_sp = np.zeros_like(htsp)
# vrt and div are initialised with zeros
vrtsp = zero_array_sp
divsp = zero_array_sp
##################################################

def dfdt(t,fn,args=None):
    # SWE are solved in vorticity divergence form
    [vrtsp, divsp, htsp] = fn
    vrt = x.spectogrd(vrtsp)
    div = x.spectogrd(divsp)
    u,v = x.getuv(vrtsp,divsp)
    ht  = x.spectogrd(htsp)

    tmp1 = u*(vrt+f)
    tmp2 = v*(vrt+f)
    tmpa, tmpb = x.getvrtdivspec(tmp1, tmp2)
    dvrtsp = -tmpb + x.grdtospec(-vrt/tau_vrt)

    tmp3 = u*(ht)
    tmp4 = v*(ht)
    tmpd, tmpe = x.getvrtdivspec(tmp3,tmp4)
    dhtsp = -tmpe + x.grdtospec( - (ht-Hbar)/tau_ht)

    tmpf = x.grdtospec(grav*ht + 0.5*(u**2+v**2))
    ddivsp = tmpa - x.lap*tmpf + x.grdtospec(-div/tau_div)

    return [dvrtsp, ddivsp, dhtsp]
###################################################

# hyper-viscosity dissipation for numerical stability
ndiss = 8.
diff_fact = (1e-5*((x.lap/x.lap[-1])**(ndiss/2)))
def diffusion(dt,F):
    hyperdiff_fact = np.exp(-dt*diff_fact)
    [vrtsp,divsp, htsp] = F
    vrtsp *= hyperdiff_fact
    divsp *= hyperdiff_fact
    htsp  *= hyperdiff_fact
    return [vrtsp, divsp, htsp]


## define integrator
stepfwd = AdamsBashforth.AdamBash(dfdt, diffusion=diffusion, ncycle=0)
tmax = 100.1*24*3600  # max time of the run
t    = 0.             # initial time, starts from 0 here
dt   = 100.           # time step, make sure it satisfies CFL condition
ii   =-1              # some index
###########################################################################
time_step = 1*24*3600 # interval at which data will be written to netcdf file, here daily data will be stored
latdim = nlats
londim = nlons
dims = [latdim,londim]
dim_names = ['latitude','longitude']

path     = '/home/suhas/Desktop/' # path where the output file would be saved
filename = 'gillIVP.nc'              # file name


fields   = ['uwind','vwind','vorticity','divergence','height'] # fields to be stored
cords    = [x.lats,x.lons]
logger = logData.logData(path+filename, fields, dim_names, dims, cords, time_step/(24*3600.), currTime= 0 , overwrite= False)
logger.createFile()

initial_name = ['htFr'] # inital fields, these are saved only once
initial_data = [S]
param_name = ['radius', 'rotation', 'gravity', 'tau_vrt', 'tau_div', 'tau_ht', 'Hbar', 'hamp', 'time_step'] # constants which a single value rather than a 2D field
param_data = [rsphere, omega, grav, tau_vrt, tau_div, tau_ht, Hbar, hamp, dt]
logger.writeInitialData(initial_name, initial_data)
logger.writeParameters(param_name, param_data)

###########################################################################
t_next = 0
while(t<tmax):
        t,[vrtsp,divsp,htsp] = stepfwd.integrate(t,[vrtsp,divsp,htsp], dt)
        if t> t_next:
            vrt = x.spectogrd(vrtsp)
            div = x.spectogrd(divsp)
            u,v = x.getuv(vrtsp,divsp)
            ht  = x.spectogrd(htsp)
            logger.writeData([u,v,vrt,div,ht]) # writing the data
            t_next += time_step

        ii = ii+1
        # Plot the fields every 200th iteration
        if np.mod(ii,200) ==0:
            print('Time in days:', t/24./3600.)
            plt.figure(1)
            plt.clf()
            plt.contourf(lonDeg,latDeg,(ht-Hbar),16,extend='both',cmap='coolwarm')
            plt.colorbar(orientation='horizontal',extend="both")
            plt.title('ht anomalies')
            plt.grid()
            plt.pause(1e-3)
