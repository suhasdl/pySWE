import numpy as np
from numpy.fft import fftfreq

class curves(object):

    def __init__(self,lat,grav,rsphere,omega,mean=0):
        self.ll    = 2*np.pi*rsphere*np.cos(lat)
        self.beta  = 2*omega*np.cos(lat)/rsphere
#        self.nPlntWave = nPlntWave
        #self.k  = 2*np.pi/self.ll
#        for wn in range(self.nPlntWave):     # planetary wave number 
#            s  = -20.*(wn-1)*2./(self.nPlntWave-1) + 20.
#        self.k  = 2*np.pi*s/self.ll
        self.kref = np.arange(-50.,50.,0.1) 
        self.k    = self.kref * (2*np.pi/self.ll)
        self.g    = grav
        self.mean = mean 
        
    # Kelvin wave
    def KW(self,Heq):
        sigma = self.k*np.sqrt(self.g*Heq) + self.mean* self.k
        freq  = (sigma*24.*60.*60.) / (2*np.pi)
        return freq,self.kref
    # Equatorial Rossby Wave
    def ERW(self,Heq,m):
        c = np.sqrt(self.g * Heq)
        de    = (2*m+1.) * self.beta/c 
        sigma = -self.beta* self.k/ (de + (self.k)**2) + self.mean* self.k
        freq  = (sigma*24.*60.*60.) / (2*np.pi)
        return freq,self.kref
    # Inertial gravity waves
    def IGW(self,Heq,m):
        c   = np.sqrt(self.g * Heq)
        sigma  = np.sqrt( (2*m+1.)*self.beta*c + (c**2)*(self.k**2) ) 
        # Some corrections ??
        for i in range(0,5):
            sigma = np.sqrt((2*m+1.)*self.beta*c + (c**2)*(self.k**2) + (c**2)*self.beta*self.k/sigma ) 
        freq  = (sigma*24.*60.*60.) / (2*np.pi)
        return freq,self.kref

    def MRG(self,Heq):
        c = np.sqrt(self.g * Heq)
        sigma = np.zeros(len(self.k))
        de  = np.sqrt( 1 + (4*self.beta)/((self.k**2)* c))
        sigma[self.k<0]  = self.k[self.k<0] * c * (0.5-0.5*de[self.k<0])
        sigma[self.k==0] = np.sqrt(c*self.beta)
        sigma[self.k>0]  = self.k[self.k>0] * c * (0.5+0.5*de[self.k>0])
        sigma = sigma + self.mean* self.k
        freq  = (sigma*24.*60.*60.) / (2*np.pi)
        return freq,self.kref


            




        
    
