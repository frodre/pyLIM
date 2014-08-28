"""
This is the first attempt at creating a linear inverse model in Python
using Greg Hakim's matlab script as a template.  I hope to eventually
turn this into a general linear inverse modelling tool

Currently performing a LIM study on surface temps to try and recreate
the findings from Newman 2014.  Script assumes that it is already in
monthly mean format.

Author: Andre Perkins
"""

import os
import sys
import numpy as np
import Stats as st
import tables as tb
from math import ceil
from scipy.io import netcdf as ncf
from scipy.signal import detrend
from time import time
from random import sample
from LIMTools import climo

#### LIM PARAMETERS ####
wsize = 12          # window size for running average
yrsize = 12         # number of elements in year
var_name = 'air'    # variable name in netcdf file
neigs = 25          # number of eof compontents to retain
num_trials = 50     # number of lim trials to run
forecast_tlim = 9   # number years to forecast
detrend_data=True   # linearly detrend the observations

# Check os, use appropriate data files
if os.name == 'nt':
    data_file = "G:/Hakim Research/data/20CR/air.2m.mon.mean.nc"
    output_loc = "G:\Hakim Research\pyLIM\LIM_data.h5"
    NCO = False  # cannot use NetCDF Ops on windows
else:
    #data_file = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
    data_file = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
    output_loc = '/home/chaos2/wperkins/data/pyLIM/LIM_data.h5'


#### LOAD DATA ####
#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
tvar = f.variables[var_name]
lats = f.variables['lat'].data
lons = f.variables['lon'].data

#account for data storage as int * scale + offset
try:
    sf = tvar.scale_factor
    offset = tvar.add_offset
    tdata = tvar.data*sf + offset
except AttributeError:
    tdata = tvar.data

#flatten t-data
spatial_shp = tdata.shape[1:]
tdata = tdata.reshape( (tdata.shape[0], np.product(spatial_shp)) )

#save data to hdf5 file
out = tb.open_file(output_loc, mode='w')
data_grp = out.create_group(out.root,
                            'data', 
                            title = 'Observations & Forecast Data',
                            filters = tb.Filters(complevel=4, complib='blosc'))
trials_grp = out.create_group(data_grp, 'trials')
obs_data = out.create_carray(data_grp, 'obs',
                                 atom = tb.Atom.from_dtype( tdata.dtype ),
                                 shape = tdata.shape,
                                 title = 'Temp Observations')
lat_data = out.create_carray(data_grp, 'lats',
                                 atom = tb.Atom.from_dtype( lats.dtype ),
                                 shape = lats.shape,
                                 title = 'Latitudes')
lon_data = out.create_carray(data_grp, 'lons',
                                 atom = tb.Atom.from_dtype( lons.dtype ),
                                 shape = lons.shape,
                                 title = 'Longitudes')
obs_data[:] = tdata
lat_data[:] = lats
lon_data[:] = lons 

#### RUN LIM ####
#Calc running mean using window size over the data
print "\nCalculating running mean..."
t1 = time()
run_mean, bedge, tedge = st.runMean(obs_data.read(), wsize, out, shaveYr=True)
t2 = time()
dur = t2 - t1
print "Done! (Completed in %f s)" % dur
obs_use = [bedge, obs_data.shape[0]-tedge]

#Calculate climatology
print "\nCalculating climatology from running mean..."
old_shp = run_mean.shape
new_shp = ( old_shp[0]/yrsize, yrsize, old_shp[1] )
obs_climo = run_mean.reshape(new_shp).sum(axis=0)/float(new_shp[0])
print "Done!"

#Remove the climo mean from the running mean and detrend
anomaly_srs = (run_mean.reshape(new_shp) - obs_climo).reshape(old_shp)
if detrend_data:
    anomaly_srs = detrend(anomaly_srs, axis=0, type='linear')
    

out_anom = out.create_carray(data_grp, 'anomaly_srs',
                             atom = tb.Atom.from_dtype(anomaly_srs.dtype),
                             shape = anomaly_srs.shape,
                             title = 'Detrended Monthly Anomaly Timeseries')
out_anom[:] = anomaly_srs

#Start running trials for LIM forecasts
fcast_times = np.array([x*yrsize for x in range(forecast_tlim+1)], dtype=np.int16)
fcast_tdim = fcast_times[-1]                   # Size of tail necessary to make forecasts
sample_tdim = old_shp[0] - fcast_tdim          # Size of useable time series for forecasting
hold_chunk = int(ceil(sample_tdim/12*0.1))     # Size(yr) of chunk to withhold for testing
test_tdim = hold_chunk*12                      # Size of testing time series
train_tdim = sample_tdim - test_tdim           # Size of training time series
fcast_shp = [num_trials*test_tdim, old_shp[1]]

test_start_idx = np.linspace(0, train_tdim-1, num_trials).astype(np.int16)
test_start_idx = np.unique(test_start_idx)
out_test_idxs = out.create_carray(data_grp, 'test_idxs',
                            atom=tb.Atom.from_dtype(test_start_idx.dtype),
                            shape=test_start_idx.shape,
                            title='Obs_run_mean Starting indicies for Test data')
out_test_idxs[:] = test_start_idx

#Create individual forecast time arrays for trials to be stored
fcast_grp = out.create_group(data_grp, 'fcast_bin')
fcast_grp._v_attrs.test_tdim = test_tdim
fcast_grp._v_attrs.yrsize = yrsize
out_fcast = [ out.create_carray(fcast_grp, 'f%i' % i, 
                                atom=tb.Atom.from_dtype(np.dtype('float32')),
                                shape = fcast_shp,
                                title = '%i Year Forecast' % i)
              for i in xrange(len(fcast_times)) ]

t1 = time()
for j,trial in enumerate(test_start_idx):
    print 'Running trial %i' % (j+1)
    
    #create training and testing set
    tsbot, tstop = (obs_use[0]+trial, obs_use[0]+trial+test_tdim)
    train_set = np.concatenate( (obs_data[(obs_use[0]-bedge):tsbot],
                                 obs_data[(tstop):obs_use[1]+tedge]),
                               axis=0 )
    
    #Calculate running mean 
    train_mean, b, t = st.runMean(train_set, wsize, shaveYr=True)
    test_mean = run_mean[trial:(trial+test_tdim+fcast_tdim)]

    #Anomalize
    old_shp = train_mean.shape
    new_shp = (old_shp[0]/yrsize, yrsize, old_shp[1])
    train_climo = climo(train_mean, yrsize)
    train_anom = (train_mean.reshape(new_shp) - train_climo).reshape(old_shp)

    old_shp = test_mean.shape
    new_shp = (old_shp[0]/yrsize, yrsize, old_shp[1])
    test_anom = (test_mean.reshape(new_shp) - train_climo).reshape(old_shp)
    
    if detrend_data:
        train_anom = detrend(train_anom, axis=0, type='linear')
        test_anom = detrend(test_anom, axis=0, type='linear')
    
    #EOFS
    print "\tCalculating EOFs..."
    eofs, eig_vals, var_pct = st.calcEOF(train_anom.T, neigs)
    print "\tLeading %i EOFS explain %f percent of the total variance" % (neigs, var_pct)
    train_eof_proj = np.dot(eofs.T, train_anom.T)
    test_eof_proj = np.dot(eofs.T, test_anom.T)
    
    # forecast for each time
    print "\tPerforming forecasts..."
    for i,tau in enumerate(fcast_times):
        x0 = train_eof_proj[:,0:train_tdim]
        xt = train_eof_proj[:,tau:(train_tdim + tau)]
        xtx0 = np.dot(xt,x0.T) 
        x0x0 = np.dot(x0, x0.T)
        M = np.dot(xtx0, np.linalg.pinv(x0x0))

        # use independent data to make forecast
        x0i = test_eof_proj[:,0:test_tdim]
        xti = test_eof_proj[:,tau:(test_tdim+tau)]
        xfi = np.dot(M, x0i)
        
        #project back into physical space and save
        start = j*test_tdim
        fin = j*test_tdim + test_tdim
        out_fcast[i][start:fin] = np.dot(eofs, xfi).T.astype(anomaly_srs.dtype)
    
t2 = time()
dur = t2 - t1
print '%i trials finished in %f s'  % (j, dur)

out.close()