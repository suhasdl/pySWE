"""
Function to test the spherical transforms in realSphTrans.py
"""
from __future__ import print_function
import realSphTrans as rsph
import numpy as np

# parameters for the simulations
nlons  = 512           # number of longitudes
ntrunc = int(nlons/3)  # spectral truncation (for alias-free computations)
nlats  = int(nlons/2)  # for gaussian grid
rsphere = 6.37122e6    # earth radius

# setup up spherical harmonic instance, set lats/lons of grid
x = rsph.realSpharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
lons,lats = np.meshgrid(x.lons, x.lats) # meshgrid

# Regrid your data to this lats and lons...

############################################################
# Sample code to calculate
# Some random fn of lats and lons. Feel free to change it...
u = np.random.randn(nlats,nlons)
v = np.random.randn(nlats,nlons)

# gradient of u
(ux,uy) = x.getgrad(u)

# divergence of u and v
div = x.getdiv(u,v)

# vorticity (curl) of u and v
vrt = x.getvrt(u,v)

# laplacian of u
ulap = x.getlap(u)

# inverse laplacian of u
uinvlap = x.getinvlap(u)
