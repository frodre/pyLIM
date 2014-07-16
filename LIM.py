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
import matplotlib.pyplot as plt
from time import time

plt.close('all')

#data_file = '/home/melt/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
data_file = '/home/melt/wperkins/data/20CR/air.2m.mon.mean.nc'
wsize = 12
var_name = 'air'

#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
tvar = f.variables[var_name]
try:
    sf = tvar.scale_factor
    offset = tvar.add_offset
    tdata = tvar.data*sf + offset
except AttributeError:
    tdata = tvar.data

#Calc running mean using window size over the data
print "\nCalculating running mean..."
t1 = time()
run_mean, bedge, tedge = st.runMean(f, var_name, wsize, num_procs=2, useNCO=True)
t2 = time()
dur = t2 - t1
print "Done! (Completed in %f s)" % dur

new_shp = [run_mean.shape[0]/wsize, wsize] + list(run_mean.shape[1:])
shp_run_mean = run_mean.reshape( new_shp )

print "\nCalculating monthly climatology from running mean..."
mon_climo = np.sum(shp_run_mean, axis=0)/float(new_shp[0])
print "Done!"

anomaly_srs = (shp_run_mean - mon_climo).reshape(([new_shp[0]*wsize] + new_shp[2:]))
x = range(len(anomaly_srs))

fig, ax = plt.subplots(2,1)
ax[0].plot(x, anomaly_srs[:,30,30])
ax[1].plot(x, run_mean[:,30,30])
fig.show()

neigs = 20

f.close()
