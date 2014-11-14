import LIM
import tables as tb
import numpy as np
import os
import scipy.io.netcdf as ncf

#if os.name == 'nt':
#    fname = "G:/Hakim Research/data/Trend_LIM_data.h5"
#else:
#    fname = '/home/chaos2/wperkins/data/pyLIM/Trend_LIM_data.h5'
#    
#f = tb.open_file(fname, 'r')
#obs = f.root.data.obs.read()

f = ncf.netcdf_file('G:/Hakim Research/data/20CR/air.2m.mon.mean.nc', 'r')
tvar = f.variables['air']
lats = f.variables['lat'].data
lons = f.variables['lon'].data
lats, lons = np.meshgrid(lats, lons, indexing='ij')

#account for data storage as int * scale + offset
try:
    sf = tvar.scale_factor
    offset = tvar.add_offset
    obs = tvar.data*sf + offset
except AttributeError:
    obs = tvar.data
    
#flatten t-data
spatial_shp = obs.shape[1:]
obs = obs.reshape( (obs.shape[0], np.product(spatial_shp)) )

yr = 12
test_dat = obs[yr:yr*13]
train_dat = np.concatenate( (obs[0:yr], obs[yr*13:]), axis=0)
sample_tdim = len(train_dat) - 9*yr
train_data = train_dat[0:sample_tdim]  #Calibration dataset

#lats = f.root.data.lats.read()
#lons = f.root.data.lons.read()
lats = lats.flatten()

test_LIM = LIM.LIM(train_data, yr, [2], 20, area_wgt_lats=lats)
out = test_LIM.forecast(test_dat, use_G1=True)

f.close()