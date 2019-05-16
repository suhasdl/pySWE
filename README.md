# pySWE
A python based framework for solving the spherical shallow water equations (SWE).
Example scripts for solving more linear Matsuno-Gill like problem as well as a non-linear turbulent case are included.
Moist shallow water code is included now.

Plot utilities include some commonly used functions used for analysing the data. 
It includes Wheeler-Kiladis plot, wave filtering, energy spectral anlaysis and so on.

Some of the results which can be generated by pySWE.

Steady state height and velocity fields for the Matsuno (vorticity) forcing case.
<img src="docs/images/height_pair.png" width="800" height = "500">

<img src="docs/images/gravity_chi.png" width="1000" height = "800">

Wheeler-Kiladis plot showing the power spectra of various tropical waves.
![](docs/images/WK_mult.png)

Momentum fluxes showing the emergence of super-rotating state.
![](docs/images/flux.png)

Kinetic energy spectra showing an inverse cascade with a scaling of -5/3.
![](docs/images/spectra.png)



## Requirements :
You need to install shtns (https://users.isterre.fr/nschaeff/SHTns/) and fftw (http://www.fftw.org/) library.
Installation guide can be found here https://users.isterre.fr/nschaeff/SHTns/compil.html
