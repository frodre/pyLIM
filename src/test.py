import LIM
import tables as tb
import numpy as np

f = tb.open_file('/home/chaos2/wperkins/data/pyLIM/Trend_LIM_data.h5', 'r')
obs = f.root.data.obs.read()

yr = 12
test_dat = obs[yr:yr*13]
train_dat = np.concatenate( (obs[0:yr], obs[yr*13:]), axis=0)
sample_tdim = len(train_dat) - 9*yr
train_data = train_dat[0:sample_tdim]  #Calibration dataset

lats = f.root.data.lats.read()
lons = f.root.data.lons.read()
lats, _ = np.meshgrid(lats, lons, indexing='ij')
lats = lats.flatten()

test_LIM = LIM.LIM(train_data, yr, [1], 20, area_wgt_lats=lats, use_G=True)
out = test_LIM.forecast(test_dat, use_G=True)