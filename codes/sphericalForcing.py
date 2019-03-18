"""
This function can be used to add random forcing at any wavenumber range desired.
Setting correlation = 0 will result in a white noise.
"""
import numpy as np
import sphTrans

class sphForcing(object):
    def __init__(self, nlons, nlats, ntrunc, rsphere, lmin, lmax, magnitude , correlation = 0.):
        self.lmin = lmin # l is spherical "total" wavenumber
        self.lmax = lmax
        self.corr = correlation    # set = 0 (for white noise)
        self.magnitude = magnitude # magnitude of forcing
        self.ntrunc = ntrunc       # typically 2/3rd of max resolved wavenumber
        self.rsphere = rsphere
        self.trans = sphTrans.Spharmt(nlons,nlats,ntrunc,rsphere,gridtype='gaussian')
        self.l = self.trans._shtns.l # total wavenumber
        self.m = self.trans._shtns.m # zonal wavenumber

        A = np.zeros(self.trans.nlm)
        A[self.l >= self.lmin] = 1.
        A[self.l >  self.lmax] = 0.
        A[self.m == 0]         = 0.     # Removing zonal mean
        self.A = A
        self.nlm = self.trans._shtns.nlm

    def forcingFn(self,F0):
        signal = self.magnitude* self.A* np.exp(np.random.rand(self.nlm)*1j*2*np.pi)
        out = (np.sqrt(1-self.corr**2))*signal + self.corr*F0
        out[self.m==0] = 0. # Remove zonal qty
        return out
