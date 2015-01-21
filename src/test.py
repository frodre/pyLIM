import LIM
import numpy as np
import tables as tb
import scipy.io.netcdf as ncf
import os

# if os.name == 'nt':
#     fname = "G:/Hakim Research/data/Trend_LIM_data.h5"
# else:
#     fname = '/home/chaos2/wperkins/data/pyLIM/Trend_LIM_data.h5'
#
# f = tb.open_file(fname, 'r')
# obs = f.root.data.obs.read()
# lats = f.root.data.lats.read()
# lons = f.root.data.lons.read()
# lats, lons = np.meshgrid(lats, lons, indexing='ij')

if os.name == 'nt':
    f = ncf.netcdf_file('G:/Hakim Research/data/20CR/air.2m.mon.mean.nc', 'r')
else:
    f = ncf.netcdf_file('/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc', 'r')

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
obs = obs.reshape((obs.shape[0], np.product(spatial_shp)))

yr = 12
test_dat = obs[0:yr*16]
train_data = np.concatenate((obs[0:yr], obs[yr*15:]), axis=0)
#sample_tdim = len(train_data) - 9*yr
#train_data = train_data[0:sample_tdim]  #Calibration dataset

#lats = f.root.data.lats.read()
#lons = f.root.data.lons.read()
lats = lats.flatten()
lons = lons.flatten()

# try:
#     h5f = tb.open_file('test.h5', mode='w')
#     h5f2 = tb.open_file('test2.h5', mode='w')
#     h5f3 = tb.open_file('test3.h5', mode='w')
#
#     test_LIM = LIM.LIM(train_data, yr, [1, 2], 20, area_wgt_lats=lats,
#                        h5file=h5f)
#     test_re_LIM = LIM.ResampleLIM(obs, yr, [1, 2], 20, 0.1, 1, area_wgt_lats=lats,
#                                   lons=lons, h5file=h5f2)
#     test_LIM.save()
#     test_LIM.save(h5f3)
#     test_re_LIM.save()
#     out1 = test_LIM.forecast(test_dat, detrend_data=True)
#     out2 = test_LIM.forecast(test_dat, detrend_data=True)
#     out3 = test_re_LIM.forecast(detrend_data=True)
# finally:
#     h5f.close()
#     h5f2.close()
#     h5f3.close()
#     f.close()

# Test that forecasts of LIM and ResampleLIM give same results on same data
# try:
#     h5f2 = tb.open_file('test2.h5', 'r')
#     h5f3 = tb.open_file('test3.h5', 'r')
#
#     reg_fcast = h5f3.root.data.fcast_bin.f1[:]
#     res_fcast = h5f2.root.data.fcast_bin.f1[:]
#
#     reg_eofs = h5f3.root.data.eofs[:]
#     res_eofs = h5f2.root.data.eofs[:]
#
#     reg = np.dot(reg_fcast.T, reg_eofs.T)
#     res = np.dot(res_fcast[0].T, res_eofs[0].T)
#
#     diff = reg - res
#     print np.abs(diff).max()
#
#     assert(reg.shape == res.shape)
#     assert(np.allclose(res_eofs[0], reg_eofs))
#     assert(np.allclose(res_fcast[0], reg_fcast))
#     assert(np.allclose(reg, res, atol=1e-6))
# finally:
#     h5f2.close()
#     h5f3.close()

try:
    ftimes = range(1, 10)  # 1 - 9 yr forecasts
    neigs = 20  # num PCs for EOFs
    hold_frac = 0.1  # fraction of data to withold for resample tests
    numTrials = 140
    h5f = tb.open_file('trials_1yr.h5', 'w')

    resample = LIM.ResampleLIM(obs, yr, [1], neigs, hold_frac, numTrials,
                               area_wgt_lats=lats,
                               lons=lons,
                               h5file=h5f)
    resample.forecast()
    resample.save()
finally:
    h5f.close()