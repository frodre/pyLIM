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
import Stats as st

data_file = '/home/melt/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
wsize = 12

#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
#tdata = f.variables['air'].data
tdata = f.variables['tas'].data#[0:1692, :, :]
dshape = tdata.shape
num_years = dshape[0]/12
shp_tdata = tdata.reshape(num_years, 12, dshape[1], dshape[2]) #reshape data

#Calc monthly climatological mean from data
print "Calculating monthly climatology..."
mon_climo = ncf.np.sum(shp_tdata, axis=0) / float(num_years)
print "Done!"

#Calc running mean using window size over the data
run_mean, bedge, tedge = st.runMean(tdata, wsize)

neigs = 20

