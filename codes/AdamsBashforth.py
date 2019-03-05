"""
Code for time stepping using a 3rd order Adams-Bashforth method.
First two steps need to be initialised through other meanns.
Here euler's forward method and 2nd order Adams-Bashforth is uded for first two timesteps.
"""
import numpy as np

class AdamBash(object):
    # you can input the hyper-viscosity diffusion function
    def __init__(self, dfdt, diffusion, ncycle = 0):
        self.ncycle = ncycle
        self.dfdt = dfdt
        self.diffusion = diffusion

    # Update fields with 3rd order adams-bashforth method
    def integrate(self, t, fields,dt, args = None):
        F = np.array(fields)
        F1 = F.copy()
        fnew = np.array(self.dfdt(t,F))
        #forward euler, then 2nd order adams-bashforth time stepping
        if self.ncycle==0:
            self.fnow = fnew.copy()
            f1 = F1 + dt*fnew   # Euler
        elif self.ncycle==1:
            f1 = F1 + dt*((3/2.)*fnew - (1/2.)*self.fnow)  #2nd order AB
        else:
            f1 = F1 + dt*((23/12.)*fnew - (16/12.)*self.fnow + (5/12.)*self.fold) #3rd order AB
            
        fq = self.diffusion(dt,f1)
        tnew = t+dt
        self.fold = self.fnow.copy()
        self.fnow = fnew.copy()
        self.ncycle += 1
        return (tnew,fq)
