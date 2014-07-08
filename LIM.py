"""
This is the first attempt at creating a linear inverse model in Python
using Greg Hakim's matlab script as a template.  I hope to eventually
turn this into a general linear inverse modelling tool

Currently performing a LIM study on surface temps to try and recreate
the findings from Newman 2014.  Script assumes that it is already in
monthly mean format.

"""

import numpy as np
from scipy.io import netcdf as ncf
from scipy.signal import convolve

data_file = '/home/melt/wperkins/data/20CR/air.2m.mon.mean.nc'
window_size = 12

#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
tdata = f.variables['air'].data
dshape = tdata.shape
num_years = dshape[0]/12
shp_tdata = tdata.reshape(num_years, 12, dshape[1], dshape[2]) #reshape data

#Calc monthly climatological mean from data
mon_climo = np.sum(shp_tdata, axis=0) / float(num_years)

#Calc running mean using window size over the data
#avg_arr = np.ones((window_size, dshape[1], dshape[2]))/float(window_size)
#run_mean = convolve(tdata, avg_arr, 'same')

neigs = 20

