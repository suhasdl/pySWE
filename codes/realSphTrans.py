"""
Wrapper class for commonly used spectral transform operations in
atmospheric models.  Provides an interface to shtns compatible
with pyspharm (pyspharm.googlecode.com).
Copied from shtns example file with slight modifications.
Unlike sphTrans.py here both input and output is in real space.
"""

import numpy as np
import shtns

class realSpharmt(object):
    def __init__(self,nlons,nlats,ntrunc,rsphere,gridtype='gaussian'):
        """initialize
        nlons:  number of longitudes
        nlats:  number of latitudes"""
        self._shtns = shtns.sht(ntrunc, ntrunc, 1, \
                shtns.sht_orthonormal+shtns.SHT_NO_CS_PHASE)
        if gridtype == 'gaussian':
            #self._shtns.set_grid(nlats,nlons,shtns.sht_gauss_fly|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
            self._shtns.set_grid(nlats,nlons,shtns.sht_quick_init|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
        elif gridtype == 'regular':
            self._shtns.set_grid(nlats,nlons,shtns.sht_reg_dct|shtns.SHT_PHI_CONTIGUOUS,1.e-10)
        self.lats = np.arcsin(self._shtns.cos_theta)
        self.lons = (2.*np.pi/nlons)*np.arange(nlons)
        self.nlons = nlons
        self.nlats = nlats
        self.ntrunc = ntrunc
        self.nlm = self._shtns.nlm
        self.degree = self._shtns.l
        self.lap = -self.degree*(self.degree+1.0).astype(np.complex)
        self.invlap = np.zeros(self.lap.shape, self.lap.dtype)
        self.invlap[1:] = 1./self.lap[1:]
        self.rsphere = rsphere
        self.lap = self.lap/self.rsphere**2
        self.invlap = self.invlap*self.rsphere**2

    def grdtospec(self,data):
        """compute spectral coefficients from gridded data"""
        return self._shtns.analys(data)

    def spectogrd(self,dataspec):
        """compute gridded data from spectral coefficients"""
        return self._shtns.synth(dataspec)

    def getuv(self,vrt,div):
        """compute wind vector from vorticity and divergence"""
        vrtspec = self._shtns.analys(vrt)
        divspec = self._shtns.analys(div)
        return self._shtns.synth((self.invlap/self.rsphere)*vrtspec,
                (self.invlap/self.rsphere)*divspec)

    def getvrtdiv(self,u,v):
        """compute vorticity and divergence from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return self._shtns.synth((self.lap*self.rsphere*vrtspec, self.lap*self.rsphere*divspec))

    def getdiv(self,u,v):
        """compute divergence from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return self._shtns.synth(self.lap*self.rsphere*divspec)

    def getvrt(self,u,v):
        """compute vorticity(curl) from wind vector"""
        vrtspec, divspec = self._shtns.analys(u, v)
        return self._shtns.synth(self.lap*self.rsphere*vrtspec)

    def getgrad(self,div):
        """compute gradient vector from spectral coeffs"""
        divspec = self._shtns.analys(div)
        vrtspec = np.zeros(divspec.shape, dtype=np.complex)
        u,v = self._shtns.synth(vrtspec,divspec)
        return u/self.rsphere, v/self.rsphere

    def getlap(self,div):
        """compute gradient vector from spectral coeffs"""
        divspec = self._shtns.analys(div)
        lapspec = self.lap * divspec
        return self._shtns.synth(lapspec)

    def getinvlap(self,div):
        """compute gradient vector from spectral coeffs"""
        divspec = self._shtns.analys(div)
        lapspec = self.invlap * divspec
        return self._shtns.synth(lapspec)
