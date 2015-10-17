__author__ = 'wperkins'

import pylim.DataTools as DT
import pylim.LIM as LIM

fname = '/home/chaos2/wperkins/data/20CR/air.2m.mon.mean.nc'
varname = 'air'

calib_obj = DT.netcdf_to_data_obj(fname, varname)
forecaster = LIM.LIM(calib_obj, 12, [1], 10)
forecaster.G_1.shape