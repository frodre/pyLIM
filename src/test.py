import LIM
import numpy as np
import tables as tb
import scipy.io.netcdf as ncf
import os

import LIMTools as Lt

# Load H5 Dataset
if os.name == 'nt':
    fname = "G:/Hakim Research/data/Trend_LIM_data.h5"
else:
    # fname = '/home/chaos2/wperkins/data/pyLIM/Trend_LIM_data.h5'
    fname = '/home/chaos2/wperkins/data/HadCRUT/HadCRUT.4.3.0.0.ens_avg_anom.h5'
    outf = '/home/chaos2/wperkins/data/pyLIM/nanstats_test.h5'

# f = tb.open_file(fname, 'r')
# obs = f.root.data.ens_avgT.read()
# lats = f.root.data.lats.read()
# lons = f.root.data.lons.read()
# lats, lons = np.meshgrid(lats, lons, indexing='ij')

# # Load netCDF dataset
# if os.name == 'nt':
#     f = ncf.netcdf_file('G:/Hakim Research/data/20CR/air.2m.mon.mean.nc', 'r')
# else:
#     f = ncf.netcdf_file('/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc', 'r')
#
# tvar = f.variables['air']
# lats = f.variables['lat'].data
# lons = f.variables['lon'].data
# lats, lons = np.meshgrid(lats, lons, indexing='ij')
#
# #account for data storage as int * scale + offset
# try:
#     sf = tvar.scale_factor
#     offset = tvar.add_offset
#     obs = tvar.data*sf + offset
# except AttributeError:
#     obs = tvar.data
#
# #flatten t-data
# spatial_shp = obs.shape[1:]
# obs = obs.reshape((obs.shape[0], np.product(spatial_shp)))
#
# # Create test/training set from obs (mimick what's in resample)
# test_dat = obs[0:yr*16]
# train_data = np.concatenate((obs[0:yr], obs[yr*15:]), axis=0)
# #sample_tdim = len(train_data) - 9*yr
# #train_data = train_data[0:sample_tdim]  #Calibration dataset

yr = 12
f = tb.open_file(fname, 'r')
obs = f.root.ens_avgT.read()
spatial_shp = obs.shape[1:]
obs = obs.reshape((obs.shape[0], np.product(spatial_shp)))
lats = f.root.lats.read()
lons = f.root.lons.read()
lats, lons = np.meshgrid(lats, lons, indexing='ij')
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

# Running resampling experiments
try:
    ftimes = range(0, 3)  # 1 - 9 yr forecasts
    neigs = 20  # num PCs for EOFs
    hold_frac = 0.1  # fraction of data to withold for resample tests
    numTrials = 30
    h5f = tb.open_file(outf, 'w')

    resample = LIM.ResampleLIM(obs[-600:], yr, ftimes, neigs, hold_frac, numTrials,
                               area_wgt_lats=lats,
                               lons=lons,
                               masked=True,
                               h5file=h5f)
    resample.forecast()
    resample.save()
finally:
    h5f.close()
    f.close()

# Testing correlation code
# try:
#     fname = '/home/chaos2/wperkins/data/pyLIM/10_trial_test.h5'
#     # fname = '/home/chaos2/wperkins/data/pyLIM/20_trial_test.h5'
#     h5f = tb.open_file(fname, 'a')
#     Lt.fcast_corr(h5f)
# finally:
#     h5f.close()

# try:
#     f1 = tb.open_file('/home/chaos2/wperkins/data/pyLIM/newman_comp_1_9fcast.h5', 'r')
#     corr1 = f1.root.stats.corr.read()[1].reshape(94, 192)
#     lats = f1.root.data.lats.read().reshape(corr1.shape)
#     lons = f1.root.data.lons.read().reshape(corr1.shape)
#     #Lt.plot_corrdata(lats, lons, corr1, title='130 Trial 1-yr LAC')
#
#     b_obs = Lt.build_trial_obs_from_h5(f1, 1)
#     b_fcast = Lt.build_trial_fcast_from_h5(f1, 1)
#
#     corr, signif = Lt.calc_corr_signif(b_fcast, b_obs)
#     corr = corr.reshape(corr1.shape)
#     signif = signif.reshape(corr1.shape)
#
#     Lt.plot_corrdata(lats, lons, corr, title='Test', signif=signif)
# finally:
#     f1.close()
