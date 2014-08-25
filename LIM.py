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
import numpy as np
import Stats as st
import tables as tb
from scipy.io import netcdf as ncf
from scipy.signal import detrend
from time import time
from random import sample

#### LIM PARAMETERS ####
wsize = 12          # window size for running average
var_name = 'air'    # variable name in netcdf file
neigs = 30          # number of eof compontents to retain
num_trials = 40     # number of lim trials to run
forecast_tlim = 144 # number months to forecast
NCO = False         # use NetCDF Operators Flag 
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
out.create_group(out.root,
                 'data', 
                 title = 'Observations & Forecast Data',
                 filters = tb.Filters(complevel=2, complib='blosc'))
obs_data = out.create_carray(out.root.data, 'obs',
                                 atom = tb.Atom.from_dtype( tdata.dtype ),
                                 shape = tdata.shape,
                                 title = 'Temp Observations')
lat_data = out.create_carray(out.root.data, 'lats',
                                 atom = tb.Atom.from_dtype( lats.dtype ),
                                 shape = lats.shape,
                                 title = 'Latitudes')
lon_data = out.create_carray(out.root.data, 'lons',
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
run_mean, bedge, tedge = st.runMean(obs_data.read(), wsize, out)
t2 = time()
dur = t2 - t1
print "Done! (Completed in %f s)" % dur

#Calculate monthly climatology
print "\nCalculating climatology from running mean..."
mon_climo = np.sum(run_mean, axis=0)
mon_climo = mon_climo/float(run_mean.shape[0])
print "Done!"

#Remove the climo mean from the running mean and detrend
anomaly_srs = run_mean - mon_climo
if detrend_data:
    anomaly_srs = detrend(anomaly_srs, axis=0, type='linear')

#Calculate EOFs
print "\nCalculating EOFs..."
t1 = time()
tmp_spat = np.copy(anomaly_srs.T, order='C')
eofs, eig_vals, var_pct = st.calcEOF(tmp_spat, neigs)
dur = time() - t1
print "Done! (Completed in %.1f s)" % dur
eof_proj = np.dot(eofs.T, tmp_spat)
del tmp_spat
print "\nLeading %i EOFS explain %f percent of the total variance" % (neigs, var_pct)

#Start running trials for LIM forecasts
time_dim = eof_proj.shape[1] - forecast_tlim
tsample = int(time_dim*0.9)
forecasts = np.zeros( [num_trials, forecast_tlim+1, neigs, (time_dim - tsample)] )
fcast_idxs = np.zeros ((num_trials, (time_dim - tsample)), dtype=np.int16)

for i in range(num_trials):
    print 'Running trial %i' % (i+1)
    # randomize sample of indices for training and testing
    rnd_idx = sample(xrange(time_dim), time_dim)
    train_idx = np.array(rnd_idx[0:tsample])
    indep_idx = np.array(rnd_idx[tsample:])
    fcast_idxs[i] = indep_idx

    # forecast for each time
    for tau in range(forecast_tlim+1):
        x0 = eof_proj[:,train_idx]
        xt = eof_proj[:,(train_idx + tau)]
        xtx0 = np.dot(xt,x0.T) 
        x0x0 = np.dot(x0, x0.T)
        M = np.dot(xtx0, np.linalg.pinv(x0x0))

        # use independent data to make forecast
        x0i = eof_proj[:,indep_idx]
        xti = eof_proj[:,(indep_idx + tau)]
        xfi = np.dot(M, x0i)
        forecasts[i, tau] = xfi

#### STORE RESULTS ####
out.create_carray(out.root.data, 'forecasts',
                  atom = tb.Atom.from_dtype( forecasts.dtype ),
                  shape = forecasts.shape,
                  title = 'EOF Space LIM Forecasts')
out.create_carray(out.root.data, 'eofs',
                  atom = tb.Atom.from_dtype( eofs.dtype ),
                  shape = eofs.shape,
                  title = 'Calculated EOFs')
out.create_carray(out.root.data, 'anomaly_srs',
                  atom = tb.Atom.from_dtype(anomaly_srs.dtype),
                  shape = anomaly_srs.shape,
                  title = 'Detrended Monthly Anomaly Timeseries')
out.create_carray(out.root.data, 'fcast_idxs',
                  atom = tb.Atom.from_dtype(fcast_idxs.dtype),
                  shape = fcast_idxs.shape,
                  title = 'Time Indices of Forecast Trial Samples')
out.close()
