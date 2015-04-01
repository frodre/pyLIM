"""
This is the first attempt at creating a linear inverse model in Python
using Greg Hakim's matlab script as a template.  I hope to eventually
turn this into a general linear inverse modelling tool

Currently performing a LIM study on surface temps to try and recreate
the findings from Newman 2014.  Script assumes that it is already in
monthly mean format.

NOTE THAT THIS SCRIPT IS NOT IN USE ANYMORE, PROBABLY DOESN'T WORK

Author: Andre Perkins
"""

import os
import numpy as np
import Stats as St
import tables as tb
from math import ceil
from scipy.io import netcdf as ncf
from scipy.signal import detrend
from time import time
from LIMTools import calc_anomaly

#### LIM PARAMETERS ####
wsize = 12          # window size for running average
yrsize = 12         # number of elements in year
var_name = 'air'    # variable name in netcdf file
neigs = 20          # number of eof compontents to retain
num_trials = 1    # number of lim trials to run
forecast_tlim = 9   # number years to forecast
detrend_data = False   # linearly detrend the observations
global_eof = True    # calculate EOFs from entire grid rather than land vs. sea

# Check os, use appropriate data files
# mask_file should contain a global grid with land points as true(1) and
#   sea points as false(0)
if os.name == 'nt':
    data_file = r"G:/Hakim Research/data/20CR/air.2m.mon.mean.nc"
    output_loc = r"G:\Hakim Research\data\pyLIM\test_LIM.h5"
    NCO = False  # cannot use NetCDF Ops on windows
else:
    #data_file = '/home/chaos2/wperkins/data/ccsm4_last_mil/tas_Amon_CCSM4_past1000_r1i1p1_085001-185012.nc'
    data_file = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
    mask_file = '/home/chaos2/wperkins/data/20CR/land.nc'
    output_loc = '/home/chaos2/wperkins/data/pyLIM/test_LIM.h5'

#### LOAD DATA ####
#Load netcdf file
f = ncf.netcdf_file(data_file, 'r')
tvar = f.variables[var_name]
lats = f.variables['lat'].data
lons = f.variables['lon'].data
lats, lons = np.meshgrid(lats, lons, indexing='ij')

#account for data storage as int * scale + offset
try:
    sf = tvar.scale_factor
    offset = tvar.add_offset
    tdata = tvar.data*sf + offset
except AttributeError:
    tdata = tvar.data
    
#flatten t-data
spatial_shp = tdata.shape[1:]
tdata = tdata.reshape((tdata.shape[0], np.product(spatial_shp)))
    
#Perform data masking if not global_eof
if not global_eof:
    f_mask = ncf.netcdf_file(mask_file)
    land_mask = f_mask.variables['land']
    
    try:
        sf = land_mask.scale_factor
        offset = land_mask.add_offset
        land_mask = land_mask.data*sf + offset
    except AttributeError:
        land_mask = land_mask.data
        
    land_mask = land_mask.squeeze().astype(np.int16).flatten()
    sea_mask = np.logical_not(land_mask)
    
    tiled_landmask = np.repeat(np.expand_dims(land_mask, 0),
                               tdata.shape[0],
                               axis=0 )
    tiled_seamask = np.repeat(np.expand_dims(sea_mask, 0),
                              tdata.shape[0],
                              axis=0)

    #Separate lat/lon coordinates into sea and land groups
    #sea_lats = np.ma.masked_array(lats, land_mask).compressed()
    #land_lats = np.ma.masked_array(lats, sea_mask).compressed()
    #sea_lons = np.ma.masked_array(lons, land_mask).compressed()
    #land_lons = np.ma.masked_array(lons, sea_mask).compressed()

#save data to hdf5 file
out = tb.open_file(output_loc, mode='w')
data_grp = out.create_group(out.root,
                            'data', 
                            title='Observations & Forecast Data',
                            filters=tb.Filters(complevel=2, complib='blosc'))
trials_grp = out.create_group(data_grp, 'trials')
obs_data = out.create_carray(data_grp, 'obs',
                             atom=tb.Atom.from_dtype(tdata.dtype),
                             shape=tdata.shape,
                             title='Temp Observations')
lat_data = out.create_carray(data_grp, 'lats',
                             atom=tb.Atom.from_dtype(lats.dtype),
                             shape=lats.shape,
                             title='Latitudes')
lon_data = out.create_carray(data_grp, 'lons',
                             atom=tb.Atom.from_dtype(lons.dtype),
                             shape=lons.shape,
                             title='Longitudes')
obs_data[:] = tdata
lat_data[:] = lats
lon_data[:] = lons 

#### RUN LIM ####
#Calc running mean using window size over the data
print "\nCalculating running mean..."
t1 = time()
run_mean, bedge, tedge = St.run_mean(obs_data.read(),
                                     wsize,
                                     out,
                                     shave_yr=True)
t2 = time()
dur = t2 - t1
print "Done! (Completed in %f s)" % dur
obs_use = [bedge, obs_data.shape[0]-tedge]

#Calculate climatology
print "\nCalculating climatology from running mean..."
old_shp = run_mean.shape
new_shp = (old_shp[0]/yrsize, yrsize, old_shp[1])
obs_climo = run_mean.reshape(new_shp).sum(axis=0)/float(new_shp[0])
print "Done!"

#Remove the climo mean from the running mean and detrend
anomaly_srs = (run_mean.reshape(new_shp) - obs_climo).reshape(old_shp)
if detrend_data:
    anomaly_srs = detrend(anomaly_srs, axis=0, type='linear')
    

out_anom = out.create_carray(data_grp, 'anomaly_srs',
                             atom=tb.Atom.from_dtype(anomaly_srs.dtype),
                             shape=anomaly_srs.shape,
                             title='Monthly Anomaly Timeseries')
out_anom[:] = anomaly_srs

#Start running trials for LIM forecasts

# fcast_tdim: Size of tail necessary to make forecasts
# sample_tdim: Size of useable time series for forecasting
# hold_chunk: Size(yr) of chunk to withhold for testing
# test_tdim: Size of testing time series
# train_tdim: Size of training time series

fcast_times = np.array([x*yrsize for x in range(forecast_tlim+1)],
                       dtype=np.int16)
fcast_tdim = fcast_times[-1]
sample_tdim = old_shp[0] - fcast_tdim
hold_chunk = int(ceil(sample_tdim/yrsize*0.1))
test_tdim = hold_chunk*yrsize
train_tdim = sample_tdim - test_tdim
fcast_shp = [num_trials*test_tdim, old_shp[1]]

test_start_idx = np.linspace(0, train_tdim-1, num_trials).astype(np.int16)
test_start_idx = np.unique(test_start_idx)
out_test_idxs = out.create_carray(
    data_grp,
    'test_idxs',
    atom=tb.Atom.from_dtype(test_start_idx.dtype),
    shape=test_start_idx.shape,
    title='Obs_run_mean Starting indicies for Test data')
out_test_idxs[:] = test_start_idx

#Create individual forecast time arrays for trials to be stored
fcast_grp = out.create_group(data_grp, 'fcast_bin')
fcast_grp._v_attrs.test_tdim = test_tdim
fcast_grp._v_attrs.yrsize = yrsize
out_fcast = [out.create_carray(fcast_grp, 'f%i' % i,
                               atom=tb.Atom.from_dtype(np.dtype('float32')),
                               shape=fcast_shp,
                               title='%i Year Forecast' % i)
             for i in xrange(len(fcast_times))]

t1 = time()
for j, trial in enumerate(test_start_idx):
    print 'Running trial %i' % (j+1)
    
    #create training and testing set
    tsbot, tstop = (obs_use[0]+trial, obs_use[0]+trial+test_tdim)
    train_set = np.concatenate((obs_data[(obs_use[0]-bedge):tsbot],
                                obs_data[tstop:obs_use[1]+tedge]),
                               axis=0)
    
    #Calculate running mean 
    train_mean, _, _ = St.run_mean(train_set, wsize, shave_yr=True)

    #Anomalize
    train_anom, climo = calc_anomaly(train_mean, yrsize)
    test_anom = run_mean[trial:(trial+test_tdim+fcast_tdim)]
    test_anom, _ = calc_anomaly(test_anom, yrsize, climo)
    
    if detrend_data:
        train_anom = detrend(train_anom, axis=0, type='linear')
    
    #Area Weight for EOF calculation
    #latg, lon_g = np.meshgrid(lats, lons, indexing='ij')
    scale = np.sqrt(np.cos(np.radians(lats)))
    wgt_train_anom = train_anom * scale.reshape(train_anom.shape[1])
    
    #EOF Calculation
    if global_eof:  # Global EOFs
        print "\tCalculating EOFs..."
        eofs, eig_vals, var_pct = St.calc_eofs(wgt_train_anom.T, neigs)
        print ("\tLeading %i EOFS explain %f percent of the total variance"
               % (neigs, var_pct))
    else:
        ocean = np.ma.masked_array(wgt_train_anom,
                                   tiled_landmask[0:len(wgt_train_anom)]
                                   ).filled(0)  # Extract masked array values
        land = np.ma.masked_array(wgt_train_anom,
                                  tiled_seamask[0:len(wgt_train_anom)]
                                  ).filled(0)
        nsea_eigs = neigs/2 + neigs % 2
        nland_eigs = neigs/2
        
        print "\tCalculating Ocean EOFs..."
        sea_eofs, sea_eigs, sea_var = St.calc_eofs(ocean.T, nsea_eigs)
        print ("""
            \t Leading %i sea EOFs explain %f percent of the total variance
               """
               % (nsea_eigs, sea_var))
                
        print "\tCalculating Land EOFs..."
        land_eofs, land_eigs, land_var = St.calc_eofs(land.T, nland_eigs)
        print ("""
            \t Leading %i land EOFs explain %f percent of the total variance
               """
               % (nland_eigs, land_var))
        
        if j == 0:        
            np.save('land_eofs.npy', land_eofs)
            np.save('sea_eofs.npy', sea_eofs)
        
        eofs = np.concatenate((sea_eofs, land_eofs), axis=1)

    # Project data into EOF space
    train_eof_proj = np.dot(eofs.T, train_anom.T)
    test_eof_proj = np.dot(eofs.T, test_anom.T)
    
    # forecast for each time
    print "\tPerforming forecasts..."
    for i, tau in enumerate(fcast_times):
        # Split dataset for different lead times
        x0 = train_eof_proj[:, 0:train_tdim]
        xt = train_eof_proj[:, tau:(train_tdim + tau)]
        xtx0 = np.dot(xt, x0.T)
        x0x0 = np.dot(x0, x0.T)
        M = np.dot(xtx0, np.linalg.pinv(x0x0))

        # use independent data to make forecast
        # TODO: I should only put in test_tdim worth of data, don't need extra
        # TODO: bit unless I'm doing direct error checks right
        # TODO: here in eof space.
        x0i = test_eof_proj[:, 0:test_tdim]
        xti = test_eof_proj[:, tau:(test_tdim+tau)]
        xfi = np.dot(M, x0i)
        
        #project back into physical space and save
        start = j*test_tdim
        fin = j*test_tdim + test_tdim
        out_fcast[i][start:fin] = np.dot(eofs, xfi).T.astype(anomaly_srs.dtype)    
            
t2 = time()
dur = t2 - t1
print '%i trials finished in %f s' % (j, dur)

out.close()