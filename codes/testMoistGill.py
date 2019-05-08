#Code to solve moist shallow water equations on a sphere
#SWE are of the vorticity-divergence form.
from __future__ import print_function
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
import seaborn
import numpy as np

seaborn.set_style('whitegrid',rc={"axes.edgecolor":'black'})
plt.ion()
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

tau_q   = 10*24*3600.   # evapouration time scale
tau_c   = 0.1*24*3600.  # condensation time scale

chi = 5500.  #### coversion factor similar to latent heat
# make sure that (h - chi*Qsat) is always positive

latDeg = np.degrees(lats)
lonDeg = np.degrees(lons)
# steady height forcing
# gaussian blob of some dimension as specified below
lats_ref   = np.radians(0)
lons_ref   = np.radians(180)
delta_lats = np.radians(10)
delta_lons = np.radians(30)
hamp = 1e-4  # amplitude of forcing
S    = hamp * np.exp(-((lons-lons_ref)/(delta_lons))**2) * np.exp(-((lats-lats_ref)/(delta_lats))**2)

######## Qsat profile ##################
lats_refQ   = np.radians(0)
lons_refQ   = np.radians(180)  # change here
delta_latsQ = np.radians(60)
delta_lonsQ = np.radians(120)
Qbar_amp = 50e-3
# gaussian in lat, lon , lat & lon and constant profiles.
# Qbar     = Qbar_amp * np.exp(-((lats-lats_refQ)/(delta_latsQ))**2)*np.exp(-((lons-np.pi)/(delta_lonsQ))**2)
#Qbar     = Qbar_amp * np.exp(-((lons-np.pi)/(delta_lonsQ))**2)
# Qbar     = Qbar_amp * np.exp(-((lats-lats_refQ)/(delta_latsQ))**2)
Qbar     = Qbar_amp * np.ones_like(lats)  # constant Qsat

# Field initialisation
zero_array = np.zeros_like(S)
htsp = x.grdtospec(zero_array + Hbar) # initial height is Hbar
zero_array_sp = np.zeros_like(htsp)
vrtsp = zero_array_sp  # vrt and div are initally assumed zero
divsp = zero_array_sp
qsp   = zero_array_sp
# here htsp is the spectral componet of ht and so on
# use spectogrd and grdtospec to transform between real and spectral domain

#######################################################################
# Eqns
def dfdt(t,fn,args=None):
    [vrtsp, divsp, htsp, qsp] = fn
    vrt = x.spectogrd(vrtsp)
    div = x.spectogrd(divsp)
    u,v = x.getuv(vrtsp,divsp)
    ht  = x.spectogrd(htsp)
    q   = x.spectogrd(qsp)
    qsp = x.grdtospec(q)
    q_condensing = q.copy()
    q_condensing[q_condensing < 0] = 0.

    tmp1 = u*(vrt+f)
    tmp2 = v*(vrt+f)
    tmpa, tmpb = x.getvrtdivspec(tmp1, tmp2)
    dvrtsp = -tmpb - x.grdtospec(vrt/tau_vrt)

    tmp3 = u*(ht)
    tmp4 = v*(ht)
    tmpd, tmpe = x.getvrtdivspec(tmp3,tmp4)
    dhtsp = -tmpe + x.grdtospec(S - (ht-Hbar)/tau_ht - chi*q_condensing/tau_c)

    tmpf = x.grdtospec(grav*ht + 0.5*(u**2+v**2))
    ddivsp = tmpa - x.lap*tmpf - x.grdtospec(div/tau_div)

    q_evap = q.copy()
    q_evap[q_evap > 0] = 0.
    qtmp1 = u*(q+Qbar)
    qtmp2 = v*(q+Qbar)
    # dq/dt = -Qbar(div.u) - u.grad(q) - H(q)/tau - (q - Qbar)/tau
    qtmp_vrtsp, qtmp_divsp = x.getvrtdivspec(qtmp1, qtmp2)
    dqsp = -qtmp_divsp + x.grdtospec( - q_condensing/tau_c - q_evap/tau_q)

    return [dvrtsp, ddivsp, dhtsp, dqsp]
###################################################
# hyperviscosity
ndiss = 8.
diff_fact = (1e-5*((x.lap/x.lap[-1])**(ndiss/2)))
l = x._shtns.l
def diffusion(dt,F):
    hyperdiff_fact = np.exp(-dt*diff_fact)
    [vrtsp,divsp, htsp, qsp] = F
    vrtsp *= hyperdiff_fact
    divsp *= hyperdiff_fact
    htsp  *= hyperdiff_fact
    qsp   *= hyperdiff_fact
    return [vrtsp, divsp, htsp, qsp]

# make sure that total moisture content is non-negative
def set_min_vapour(qp,qbar):
    qtot = qp + qbar
    qtot[qtot<0] = 0
    return (qtot-qbar)
##################################################################
# setting up the integration
stepfwd = AdamsBashforth.AdamBash(dfdt,diffusion, ncycle=0)
tmax = 10.01*24*3600  #(time to integrate, here 10 days)
t    = 0. # initial time = 0
dt   = 50. # timestep = 50 or 100sec (make sure it satisfies CFL)
ii   = -1  # some index for plotting
t_next = 0
################################################################
# Logging the data
time_step = 1*24*3600  # interval for logging the data
# here the data is logged at the interval of 1 day
latdim = nlats
londim = nlons
dims = [latdim,londim]
dim_names = ['latitude','longitude']

path     = '/home/suhas/Desktop/'  # path to save the file
filename = 'testMoist.nc' # filename

fields   = ['uwind','vwind','vorticity','divergence','height','qprime']
 # variables to store (this only saves the data for a field which is a fn of lat, lon and time).
cords    = [x.lats,x.lons]
logger = logData.logData(path+filename, fields, dim_names, dims, cords, time_step/(24*3600.), currTime= 0 , overwrite= False)
logger.createFile()

initial_name = ['Qbar','htFr'] # use for saving initial profiles. This is a fn of lat and lon but not time.
initial_data = [Qbar,S]
const_name = ['radius','rotation','gravity', 'chi'] #for constants
const_data = [rsphere,omega,grav, chi]
param_name = ['tau_vrt','tau_div','tau_ht','tau_c','tau_q','Hbar','hamp','time_step']
param_data = [tau_vrt  , tau_div , tau_ht , tau_c , tau_q,  Hbar , hamp , dt ]
logger.writeInitialData(initial_name, initial_data)
logger.writeParameters((param_name+const_name), (param_data+const_data))
#################################################

while(t<tmax):

        t,[vrtsp,divsp,htsp,qsp] = stepfwd.integrate(t,[vrtsp,divsp,htsp,qsp], dt)
        q = x.spectogrd(qsp)
        q = set_min_vapour(q,Qbar)
        qsp = x.grdtospec(q)

        if t> t_next:
            vrt = x.spectogrd(vrtsp)
            div = x.spectogrd(divsp)
            u,v = x.getuv(vrtsp,divsp)
            ht = x.spectogrd(htsp)
            q = x.spectogrd(qsp)
            logger.writeData([u,v,vrt,div,ht,q])
            t_next += time_step

        ii = ii+1

        if np.mod(ii,100) ==0:
            print ('Time in days:', t/24./3600.)

            plt.figure(1)
            plt.clf()
            maxAnomaly = np.amax(abs(q))
            if maxAnomaly == 0 : maxAnomaly = 0.0001
            levels = np.linspace(-maxAnomaly, maxAnomaly, 16)
            plt.contourf(lonDeg,latDeg,q,levels,extend='both',cmap='coolwarm')
            plt.colorbar(orientation='horizontal',extend="both")
            plt.title('q')
            # plt.contour(lonDeg, latDeg, Qbar,6,colors='r')
            plt.grid()
            plt.pause(1e-3)

            plt.figure(2)
            plt.clf()
            maxAnomaly = np.amax(abs(ht-300))
            levels = np.linspace(-maxAnomaly, maxAnomaly, 16)
            plt.contourf(lonDeg,latDeg,ht-300.,levels,extend='both',cmap='coolwarm')
            plt.colorbar(orientation='horizontal',extend="both")
            plt.title('ht')
            # plt.contour(lonDeg, latDeg, Qbar,6,colors='r')
            plt.grid()
            plt.pause(1e-3)
