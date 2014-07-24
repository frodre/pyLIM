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
from scipy.sparse.linalg import eigs
import Stats as st
import matplotlib.pyplot as plt
from time import time
from random import sample
import os

#Clear all previous plots
plt.close('all')

wsize = 12
var_name = 'air'
neigs = 20
num_trials = 1
NCO = True #NetCDF Operators Flag, always false on Windows for now

if os.name == 'nt':
    data_file = "G:/Hakim Research/data/20CR/air.2m.mon.mean.nc"
    NCO = False
else:
    #data_file = '/home/melt/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
    data_file = '/home/melt/wperkins/data/20CR/air.2m.mon.mean.nc'

#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
tvar = f.variables[var_name]
#account for data storage as int * scale + offset
try:
    sf = tvar.scale_factor
    offset = tvar.add_offset
    tdata = tvar.data*sf + offset
except AttributeError:
    tdata = tvar.data

#Calc running mean using window size over the data
print "\nCalculating running mean..."
t1 = time()
run_mean, bedge, tedge = st.runMean(f, var_name, wsize, num_procs=2, useNCO=NCO)
t2 = time()
dur = t2 - t1
print "Done! (Completed in %f s)" % dur

new_shp = [run_mean.shape[0]/wsize, wsize] + list(run_mean.shape[1:])
shp_run_mean = run_mean.reshape( new_shp )

print "\nCalculating monthly climatology from running mean..."
mon_climo = np.sum(shp_run_mean, axis=0)/float(new_shp[0])
print "Done!"

anomaly_srs = (shp_run_mean - mon_climo).reshape(([new_shp[0]*wsize] + new_shp[2:]))

#Reshape data for covariance
shp = anomaly_srs.shape
shp_anomaly = anomaly_srs.reshape(shp[0], shp[1]*shp[2])
print "\nCalculating EOFs..."
t1 = time()
eofs, eig_vals, var_pct = st.calcEOF(shp_anomaly.T, neigs)
dur = time() - t1
print "Done! (Completed in %.1f s)" % dur
eof_proj = np.dot(eofs.T, shp_anomaly.T)

print "\nLeading %i EOFS explain %f percent of the total variance" % (neigs, var_pct)

#Start running trials for LIM forecasts
for i in range(num_trials):
    print """
=========
Trial %i
=========
    """ % i
    #randomize sample of indices for training and testing
    tlimit = 12*12
    time_dim = eof_proj.shape[1] - tlimit
    tsample = int(time_dim*0.9)
    rnd_idx = sample(xrange(time_dim), time_dim)

    for tau in range(tlimit):
        train_idx = np.array(rnd_idx[0:tsample])
        x0 = eof_proj[:,train_idx]
        xt = eof_proj[:,(train_idx + tau)]
        xtx0 = np.dot(xt,x0.T) 
        x0x0 = np.dot(x0, x0.T)
        M = np.dot(xtx0, np.linalg.pinv(x0x0))

        #Test on training data
        xf = np.dot(M, x0)
        err = xf - xt
        if (tau + 1) % 12 == 0:
            print 'Mean training set error: %f' % abs(err).mean()

        #Test on independent data
        indep_idx = np.array(rnd_idx[(tsample+1):])
        x0i = eof_proj[:,indep_idx]
        xti = eof_proj[:,(indep_idx + tau)]
        xfi = np.dot(M, x0i)
        erri = xfi - xti

        if (tau + 1) % 12 == 0:
            print 'Mean training set error: %f' % abs(err).mean()
            print 'Mean testing set error: %f' % abs(erri).mean()
         
f.close()
