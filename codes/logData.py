from netCDF4 import Dataset
import numpy as np
from __future__ import print_function

class logData(object):

    """
    This class is a wrapper over the netCDF4 library to simplify
    writing fields to a nc file
    """

    def __init__(self, filename, fieldnames, dimNames, dims, cords, timeStep, currTime, \
            overwrite=False):

        assert len(dimNames) == len(dims), \
            "number of dimensions must match dimension names"

        self.filename = filename
        self.fields = fieldnames
        self.dims = dims
        self.currTime = currTime
        self.timeStep = timeStep
        self.lats = cords[0]
        self.lons = cords[1]
        self.dims = dims
        self.dimNames = dimNames

    def createFile(self):
        ncFile = Dataset(self.filename, 'w', clobber=False)

        # create a time dimension
        if 'time' not in self.dimNames:
            ncFile.createDimension('time', None)
        # Create dimensions
        for i in range(len(self.dims)):
            ncFile.createDimension(self.dimNames[i], self.dims[i])
        # Create variables
        ncFile.createVariable('time', 'f4', ('time',))

        for i in range(len(self.fields)):
            ncFile.createVariable(self.fields[i],'f8', list(ncFile.dimensions.keys()))

        ncFile.createVariable('latitude', 'f8', (self.dimNames[0],))
        ncFile.createVariable('longitude', 'f8',(self.dimNames[1],))
        ncFile.variables['latitude'][:] = self.lats
        ncFile.variables['longitude'][:] = self.lons
        ncFile.description = 'Simulation data'
        self.ii = 0
        ncFile.close()
        print('Created file ' + self.filename)



    def resumeFile(self):
        ncFile = Dataset(self.filename, 'r+')
        tm = ncFile['time'][:]
        tmp = np.where(tm == (self.currTime-self.timeStep))
        self.ii = np.int(tmp[0]) + 1
        print('Resumed file ' + self.filename)
        ncFile.close()
####################################################################################

    def writeData(self, fields):
        ncFile = Dataset(self.filename, 'r+')
        assert len(fields) == len(self.fields), \
            "all fields must be written at the same time."

        j = self.ii
        t = self.currTime
        print('Writing data at time: ', t)

        variable = list(ncFile.variables.keys())
        variable.remove('time')
        variable.remove('latitude')
        variable.remove('longitude')
        ncFile.variables['time'][j] = t

        for i in range(len(variable)):
            temp = ncFile.variables[variable[i]]
            temp[j,:] = fields[i]

        self.currTime += self.timeStep
        self.ii +=1
        ncFile.close()


    def finishLogging(self):
        print('Finished logging data to ', self.filename)

#########################################################
    def writeInitialData(self,names,field):
        ncFile = Dataset(self.filename, 'r+')
        assert len(names) == len(field), \
        "all fields must be written at the same time."

        print('Writing Initial Data')
        dims = list(ncFile.dimensions.keys())
        dims = dims[1:]
        for i in range(len(names)):
            path = '/initial/'
            temp = ncFile.createVariable(path+names[i],'f8',dims)
            temp[:] = field[i]
        ncFile.close()

# Use parameters to write the constants of the simulation.
# Use it to store the fields which are not a function of time.
    def writeParameters(self,names,values):
        ncFile = Dataset(self.filename, 'r+')
        assert len(names) == len(values), \
        "all fields must be written at the same time."
        print('Writing parameters')
        for i in range(len(names)):
            path = '/parameters/'
            temp = ncFile.createVariable(path+names[i],'f8')
            temp[:] = values[i]
        ncFile.close()
#############################################################
